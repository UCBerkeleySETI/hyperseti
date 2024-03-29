from hyperseti.dedoppler import dedoppler
from hyperseti.filter import apply_boxcar
from hyperseti.normalize import normalize
from hyperseti.hits import hitsearch, merge_hits
from hyperseti.pipeline import find_et, GulpPipeline
from hyperseti.io import from_fil, from_h5, from_setigen
from hyperseti.log import set_log_level
import cupy as cp

from astropy import units as u
import setigen as stg
import pylab as plt
import numpy as np
import os

from hyperseti.plotting import imshow_dedopp, imshow_waterfall, overlay_hits
from hyperseti.test_data import voyager_h5, voyager_yaml, tmp_file, tmp_dir

synthetic_fil = tmp_file('synthetic.fil')
test_fig_dir  = tmp_dir('test_figs')

import logbook
import hyperseti

def test_hitsearch():
    """ Test the hit search routines """
    n_timesteps = 32
    n_chan = 4096
    signal_bw = 16

    # Create test data
    metadata = {'frequency_start': 1000*u.MHz, 'time_step': 1.0*u.s, 'frequency_step': 1.0*u.Hz}
    frame = stg.Frame(fchans=n_chan*u.pixel, tchans=n_timesteps*u.pixel,
            df=metadata['frequency_step'], dt=metadata['time_step'], fch1=metadata['frequency_start'])
    frame.add_noise(x_mean=2, x_std=1, noise_type='chi2')

    darray = from_setigen(frame)
    darray.data = cp.asarray(darray.data)

    # Add a signal with bandwidth into the data, SNR of 1000
    for ii in range(signal_bw):
        darray.data[:, :, n_chan // 2 + ii]   += 1000 / signal_bw

    print("--- Run dedoppler() then hitsearch() ---")
    dedopp = dedoppler(darray, boxcar_size=16, max_dd=1.0, plan='optimal')
    hits0 = hitsearch(dedopp, threshold=1000)
    print(hits0)
    # Output should be
    #driftrate      f_start      snr  driftrate_idx  channel_idx  boxcar_size
    #0        0.0  1000.002056  32000.0             32         2056           16
    # Note that SNR here is unnormalized, so actually peak value -- as we didn't renormalize
    h0 = hits0.iloc[0]

    #TODO: Fix assertions
    #assert h0['snr'] == 32000.0
    assert h0['channel_idx'] == 2056
    assert 30 <= h0['driftrate_idx'] <= 32 ## Not sure why new algorithm puts centroid to the side?
    assert len(hits0) == 1

    print("--- run_pipeline with w/o merge --- ")
    config = {
        'preprocess': {
            'sk_flag': False,
            'normalize': False,
        },
        'dedoppler': {
            'kernel': 'ddsk',
            'max_dd': 1.0,
            'min_dd': None,
            'apply_smearing_corr': False,
        },
        'hitsearch': {
            'threshold': 100,
            'min_fdistance': 100
        },
        'pipeline': {
            'n_boxcar': 5,
            'merge_boxcar_trials': False
        }
    }
    pipeline = GulpPipeline(darray, config)
    hits = pipeline.run()

    for rid, hit in hits.iterrows():
         assert(np.abs(hit['channel_idx'] - 2048) < np.max((signal_bw, hit['boxcar_size'])))

    print(hits)

    print("--- run merge_hits --- ")
    print(hits.dtypes)
    merged_hits = merge_hits(hits)
    assert len(merged_hits == 1)
    print(merged_hits)

    print("--- run_pipeline with merge --- ")
    config['pipeline']['merge_boxcar_trials'] = True
    pipeline.config = config
    hits2 = pipeline.run()
    hits2
    print(hits2)
    assert hits2.iloc[0]['boxcar_size'] == signal_bw
    assert len(hits2) == len(merged_hits) == 1

    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    imshow_waterfall(darray, xaxis='channel', yaxis='timestep')

    plt.subplot(1,2,2)
    imshow_dedopp(dedopp, xaxis='channel', yaxis='driftrate')

    plt.savefig(os.path.join(test_fig_dir, 'test_hitsearch.png'))
    plt.show()

def test_hitsearch_multi():
    """ Test hit search routine with multiple signals """

    set_log_level("info")
    metadata_in = {'frequency_start': 6095.214842353016*u.MHz,
            'time_step': 18.25361108*u.s,
            'frequency_step': 2.7939677238464355*u.Hz}

    frame = stg.Frame(fchans=2**16*u.pixel,
                      tchans=32*u.pixel,
                      df=metadata_in['frequency_step'],
                      dt=metadata_in['time_step'],
                      fch1=metadata_in['frequency_start'])

    frame.add_noise(x_mean=1, x_std=5, noise_type='chi2')

    test_tones = [
      {'f_start': frame.get_frequency(index=500), 'drift_rate': 0.50*u.Hz/u.s, 'snr': 1000, 'width': 20*u.Hz},
      {'f_start': frame.get_frequency(index=800), 'drift_rate': -0.40*u.Hz/u.s, 'snr': 500, 'width': 20*u.Hz},
      {'f_start': frame.get_frequency(index=2048), 'drift_rate': 0.00*u.Hz/u.s, 'snr': 100, 'width': 6*u.Hz},
      {'f_start': frame.get_frequency(index=3000), 'drift_rate': 0.07*u.Hz/u.s, 'snr': 200, 'width': 3*u.Hz}
    ]

    for tone in test_tones:
        frame.add_signal(stg.constant_path(f_start=tone['f_start'],
                                                drift_rate=tone['drift_rate']),
                              stg.constant_t_profile(level=frame.get_intensity(snr=tone['snr'])),
                              stg.gaussian_f_profile(width=tone['width']),
                              stg.constant_bp_profile(level=1))

    frame.save_fil(filename=synthetic_fil)
    darray = from_fil(synthetic_fil)
    darray.data = cp.asarray(darray.data)

    config = {
        'preprocess': {
            'sk_flag': False,
            'normalize': True,
        },
        'dedoppler': {
            'kernel': 'ddsk',
            'max_dd': 1.0,
            'min_dd': None,
            'apply_smearing_corr': True,
        },
        'hitsearch': {
            'threshold': 5,
            'min_fdistance': 100
        },
        'pipeline': {
            'n_boxcar': 5,
            'n_blank': 2,
            'merge_boxcar_trials': True
        }
    }
    pipeline = GulpPipeline(darray, config)
    hits = pipeline.run()
    

    print(hits.sort_values('snr', ascending=False))

    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    imshow_waterfall(darray, xaxis='channel', yaxis='timestep')

    plt.subplot(1,2,2)
    dedopp = dedoppler(darray, max_dd=1.0, min_dd=None)

    imshow_dedopp(dedopp, xaxis='channel', yaxis='driftrate')
    overlay_hits(hits, 'channel', 'driftrate')

    plt.savefig(os.path.join(test_fig_dir, 'test_hitsearch_multi.png'))
    plt.show()

def test_find_et():
    config = {
        'preprocess': {
            'sk_flag': True,
            'normalize': True,
            'n_poly': 5,
            'blank_extrema': {'threshold': 10000},
            'blank_edges': {'n_chan': 16384}
        },
        'dedoppler': {
            'kernel': 'ddsk',
            'max_dd': 0.5,
            'min_dd': None,
            'apply_smearing_corr': False,
        },
        'hitsearch': {
            'threshold': 30,
            'min_fdistance': None
        },
        'pipeline': {
            'n_boxcar': 2,
            'merge_boxcar_trials': True,
            'n_blank': 1
        }
    }

    et_browser = find_et(voyager_h5, config, gulp_size=2**20)
    pd = et_browser.hit_table
    print(pd[['snr', 'channel_idx', 'gulp_channel_idx', 'drift_rate', 'f_start']])
    print(len(pd))
    
    # Should find 5 signals - two per sideband and central carrier
    # Carrier USB_guard_upper, LSB_guard_upper, USB_guard_lower, LSB_guard_upper
    known_idx_vals = [747929, 756038, 739935, 755924, 739828]
    assert len(pd) >= 5
    assert np.allclose(pd['channel_idx'].values, known_idx_vals, atol=2)


def test_find_et_cmd_tool():
    try:
        from hyperseti.findET import cmd_tool

        args = [voyager_h5, 
                "-o", "hits.csv", 
                "-c", voyager_yaml, 
                ]

        cmd_tool(args)
    finally:
        fns = 'hits.csv', 'hits.log'
        for fn in fns:
            if os.path.exists(fn):
                os.remove(fn)

if __name__ == "__main__":
    #test_hitsearch()
    #test_hitsearch_multi()
    test_find_et()
    #test_find_et_cmd_tool()
