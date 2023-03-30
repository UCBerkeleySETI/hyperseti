from hyperseti.blanking import blank_edges, blank_extrema, blank_hits, blank_hits_gpu
from hyperseti.io import from_setigen
from hyperseti.pipeline import GulpPipeline

import setigen as stg
from astropy import units as u
import cupy as cp
import numpy as np

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
        },
        'sk_flag': {
            'n_sigma': 1,
        },
        'dedoppler': {
            'kernel': 'ddsk',
            'max_dd': 4.0,
            'min_dd': None,
            'apply_smearing_corr': False
        },

        'hitsearch': {
            'threshold': 25,
            'min_fdistance': None
        },
        'pipeline': {
            'n_boxcar': 4,
            'merge_boxcar_trials': True
        }
    }
    pipeline = GulpPipeline(d, config)
    df = pipeline.run()
    db = blank_hits(d, df)
    db_data0 = cp.copy(db.data)

    for idx in start_idxs:
        assert db.data[0,0, idx] == 0
    print("Hits blanked!")

    # Try again with gpu kernel version 
    d = generate_data_array_multihits()
    d = d.sel({'frequency': slice(0, 2**12)}, space='gpu')

    pipeline = GulpPipeline(d, config)
    df = pipeline.run()
    db = blank_hits_gpu(d, df)

    for idx in start_idxs:
        assert db.data[0,0, idx] == 0
    print("Hits blanked!")


if __name__ == "__main__":
    test_blank_hits()
    test_blank_edges()
    test_blank_extrema()
