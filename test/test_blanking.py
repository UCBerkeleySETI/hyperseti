from hyperseti.blanking import blank_edges, blank_extrema, blank_hits, blank_hits_gpu
from hyperseti.kernels.blank_hits import BlankHitsMan
from hyperseti.io import from_setigen
from hyperseti.pipeline import GulpPipeline

from hyperseti.normalize import normalize
from hyperseti.kurtosis import sk_flag
from hyperseti.hits import hitsearch
from hyperseti.dedoppler import dedoppler
from hyperseti.io import from_h5
from hyperseti.hit_browser import HitBrowser

import setigen as stg
from astropy import units as u
import cupy as cp
import numpy as np

from hyperseti.test_data import voyager_fil, voyager_h5

def generate_data_array_multihits():
    metadata = {'fch1': 6095.214842353016*u.MHz, 
                'dt': 18.25361108*u.s, 
                'df': 2.7939677238464355*u.Hz}

    frame = stg.Frame(fchans=2**12*u.pixel,
                    tchans=32*u.pixel,
                    df=metadata['df'],
                    dt=metadata['dt'],
                    fch1=metadata['fch1'])

    start_idxs = 500, 700, 2048, 3000
    test_tones = [
    {'f_start': frame.get_frequency(index=start_idxs[0]), 'drift_rate': 0.70*u.Hz/u.s, 'snr': 100, 'width': 20*u.Hz},
    {'f_start': frame.get_frequency(index=start_idxs[1]), 'drift_rate': -0.55*u.Hz/u.s, 'snr': 100, 'width': 20*u.Hz},
    {'f_start': frame.get_frequency(index=start_idxs[2]), 'drift_rate': 0.00*u.Hz/u.s, 'snr': 40, 'width': 6*u.Hz},
    {'f_start': frame.get_frequency(index=start_idxs[3]), 'drift_rate': 0.07*u.Hz/u.s, 'snr': 50, 'width': 3*u.Hz}
    ]

    noise = frame.add_noise(x_mean=10, x_std=5, noise_type='chi2')

    for tone in test_tones:
        signal = frame.add_signal(stg.constant_path(f_start=tone['f_start'],
                                                drift_rate=tone['drift_rate']),
                            stg.constant_t_profile(level=frame.get_intensity(snr=tone['snr'])),
                            stg.gaussian_f_profile(width=tone['width']),
                            stg.constant_bp_profile(level=1))

    d = from_setigen(frame)
    return d


def generate_data_array():
    frame = stg.Frame(fchans=8192*u.pixel, tchans=32*u.pixel, 
                      df=2.79*u.Hz, dt=18.253611008*u.s,
                      fch1=1000.0*u.MHz)
    noise = frame.add_noise(x_mean=10, noise_type='chi2')
    signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(index=200),
                                                drift_rate=2*u.Hz/u.s),
                            stg.constant_t_profile(level=frame.get_intensity(snr=1000000)),
                            stg.gaussian_f_profile(width=10*u.Hz),
                            stg.constant_bp_profile(level=1))
    darr =  from_setigen(frame)
    darr.data = cp.asarray(darr.data)
    return darr

def test_blank_edges():
    darr = generate_data_array()
    darr = blank_edges(darr, 128)
    assert np.allclose(darr.data[:, 0, :128], 0)
    assert np.allclose(darr.data[:, 0, :128], 0)

def test_blank_extrema():
    darr = generate_data_array()
    darr = blank_extrema(darr, 1000) 
    assert np.max(darr) < 1000

def test_blank_hits():
    d = generate_data_array_multihits()
    start_idxs = 500, 700, 2048, 3000
    d = d.sel({'frequency': slice(0, 2**12)}, space='gpu')

    config = {
        'preprocess': {
            'sk_flag': False,
            'normalize': True,
            'poly_fit': 5,
        },
        'sk_flag': {
            'n_sigma': 5,
        },
        'dedoppler': {
            'kernel': 'ddsk',
            'max_dd': 4.0,
            'min_dd': None,
            'apply_smearing_corr': False
        },

        'hitsearch': {
            'threshold': 20,
            'min_fdistance': 50
        },
        'pipeline': {
            'n_boxcar': 5,
            'merge_boxcar_trials': True
        }
    }
    pipeline = GulpPipeline(d, config)
    df = pipeline.run()
    db = blank_hits(d, df)
    print(df[['snr', 'channel_idx', 'gulp_channel_idx', 'drift_rate']])
    db_data0 = cp.copy(db.data)
    print(df.columns)

    for idx in start_idxs:
        assert db.data[0,0, idx] == 0
    print("Hits blanked!")

    # Try again with gpu kernel version 
    d = generate_data_array_multihits()
    d = d.sel({'frequency': slice(0, 2**12)}, space='gpu')

    pipeline = GulpPipeline(d, config)
    df = pipeline.run()
    print(df[['snr', 'channel_idx', 'gulp_channel_idx', 'drift_rate']])

    db = blank_hits_gpu(d, df)

    for idx in start_idxs:
        assert db.data[0,0, idx] == 0
    print("Hits blanked!")

    # Test with kernel manager
    bm = BlankHitsMan()
    d = generate_data_array_multihits()
    d = d.sel({'frequency': slice(0, 2**12)}, space='gpu')
    pipeline = GulpPipeline(d, config)
    df = pipeline.run()

    db = blank_hits_gpu(d, df, mm=bm)
    print(bm.info())



def test_voyager_blanking():
    """ Test blanking on main DC spike """

    darr = from_h5(voyager_h5)
    darr = darr.sel({'frequency': slice(0, 2**20)})
    darr.data = cp.asarray(darr.data)

    flags = sk_flag(darr)
    darr = normalize(darr, flags)
    dd = dedoppler(darr, max_dd=1)
    hits = hitsearch(dd, threshold=50)
    darr = blank_hits_gpu(darr, hits)

    hit_browser = HitBrowser(darr, hits)
    hx = hit_browser.extract_hit(0, padding=100)
    print(hit_browser.hit_table[['snr', 'channel_idx', 'gulp_channel_idx', 'drift_rate']])

    assert cp.asnumpy(hx.data[..., 100]).sum() == 0
    assert cp.asnumpy(hx.data[..., 50]).sum() > 0
    print("Single gulp passed!")

    # Test with offset 
    # Read middle [2^18] -> [2^18 2^18] <- [2^18]
    darr = from_h5(voyager_h5)
    darr = darr.sel({'frequency': slice(2**18, 3*2**18)})
    darr.data = cp.asarray(darr.data)

    flags = sk_flag(darr)
    darr = normalize(darr, flags)
    dd = dedoppler(darr, max_dd=1)
    hits = hitsearch(dd, threshold=50)
    darr = blank_hits_gpu(darr, hits)

    # DC BIN should be ZERO
    print(darr.data[..., 262144].T)
    print(darr.data[0, 0, 262144-6:262144+6+1])

    for hit in hits.iterrows():
        row_idx, htbl = hit
        gid = htbl['gulp_channel_idx']
        assert(darr.data[0, 0, int(gid)].sum() == 0)
    print("Offset gulp passed!")


if __name__ == "__main__":
    test_blank_hits()
    test_blank_edges()
    test_blank_extrema()
    test_voyager_blanking()
