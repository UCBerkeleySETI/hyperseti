from hyperseti import logger, dedoppler, run_pipeline, apply_boxcar, normalize, merge_hits, hitsearch

import cupy as cp
import logging
from astropy import units as u
import setigen as stg
import pylab as plt
import pandas as pd
import numpy as np

from plotting import imshow_dedopp, imshow_waterfall, overlay_hits

def test_dedoppler():
    """ Basic tests of the dedoppler functionality """
    
    # zero drift test, no normalization
    test_data = np.ones(shape=(32, 1024))
    test_data[:, 511] = 10
    
    metadata = {'fch1': 1000*u.MHz, 
            'dt': 1.0*u.s, 
            'df': 1.0*u.Hz}
    
    dedopp, metadata = dedoppler(test_data, metadata, boxcar_size=1,
                                 max_dd=1.0)
    
    assert np.max(dedopp) == np.sum(test_data[:, 511])
    

    
    for dr_test in (0.0, 0.1, 0.5, -0.25, -0.5):
        # single drifting tone
        frame = stg.Frame(fchans=2**10*u.pixel, tchans=32*u.pixel,
                  df=metadata['df'], dt=metadata['dt'], fch1=metadata['fch1'])
    
        tone = {'f_start': frame.get_frequency(index=500), 'drift_rate': dr_test * u.Hz / u.s, 'snr': 500, 'width': metadata['df']}
        noise = frame.add_noise(x_mean=1, noise_type='chi2')

        signal = frame.add_signal(stg.constant_path(f_start=tone['f_start'],
                                                    drift_rate=tone['drift_rate']),
                                  stg.constant_t_profile(level=frame.get_intensity(snr=tone['snr'])),
                                  stg.gaussian_f_profile(width=tone['width']),
                                  stg.constant_bp_profile(level=1))

        dedopp, metadata = dedoppler(frame.data, metadata, boxcar_size=1,
                                     max_dd=1.0)

        # Manual dedoppler search -- just find max channel (only works if S/N is good)
        manual_dd_tot = 0
        for ii in range(frame.data.shape[0]):
            manual_dd_tot += np.max(frame.data[ii])
        imshow_dedopp(dedopp, metadata, show_colorbar=False)

        maxpixel = np.argmax(dedopp)
        mdrift, mchan = (maxpixel // 1024, maxpixel % 1024)
        optimal_drift = metadata['drift_trials'][mdrift]
        maxpixel_val = np.max(dedopp)
        
        frac_recovered = (maxpixel_val / manual_dd_tot)
        
        print(f"Inserted drift rate:  {tone['drift_rate']} \tSUM: {manual_dd_tot:2.2f}")
        print(f"Recovered drift rate: {optimal_drift} Hz / s \tSUM: {maxpixel_val:2.2f}\n")
        
        # Channel should detected at +/- 1 chan
        assert np.abs(mchan - 500) <= 1
        
        # Drift rate should be detected +/- 1 drift resolution
        assert np.abs(optimal_drift - dr_test) <= 1.01*metadata['dd'].value

        # Recovered signal sum should be close to manual method
        assert 1.001 >= frac_recovered >= 0.825
    
    # Finish off figure plotting
    plt.colorbar()
    plt.savefig('docs/figs/test_dedoppler.png')
    plt.show()

def test_dedoppler_boxcar():
    def generate_drifting_tone(n_chan, n_timesteps, n_drift_per_step, sigval=10):
        """ Simple tone generator to generate smeared tones """
        bg = np.zeros((n_timesteps, n_chan))

        for ii in range(0, bg.shape[0]):
            for nd in range(n_drift_per_step):
                z = n_drift_per_step * ii + nd
                bg[ii, bg.shape[1]//2 + z] = sigval / n_drift_per_step
        return bg

    def maxhold_dedoppler(data):
        """ A simple and crappy dedoppler algorithm 

        Finds the top value in each timestep and adds together.
        This method only works for single channel tones with high SNR
        """
        manual_dd_tot = 0
        for ii in range(bg.shape[0]):
            manual_dd_tot += np.max(bg[ii])
        return manual_dd_tot

    # Drift rate of 2 channels / integration
    # To simulate channel smearing
    metadata = {'fch1': 1000*u.MHz, 'dt': 1.0*u.s, 'df': 1.0*u.Hz}

    bg = generate_drifting_tone(n_chan=256, n_timesteps=32, n_drift_per_step=2, sigval=10)
    print(f"Total power in frame: {np.sum(bg)}")

    # Compute dedoppler using basic maxhold method
    maxhold_res = maxhold_dedoppler(bg)
    print(f"MAXHOLD recovered power: {maxhold_res}")

    # With boxcar_size = 1 we should recover 160
    dedopp, metadata = dedoppler(bg, metadata, boxcar_size=1, 
                                 max_dd=2.0)

    maxpixel = np.argmax(dedopp)
    mdrift, mchan = (maxpixel // 1024, maxpixel % 1024)
    maxpixel_val = np.max(dedopp)
    print(f"dedopp recovered power (boxcar 1): {maxpixel_val}")
    assert maxpixel_val == maxhold_res

    # With boxcar_size = 2 we should recover 320 (full amount)
    metadata = {'fch1': 1000*u.MHz, 'dt': 1.0*u.s, 'df': 1.0*u.Hz}
    dedopp, metadata = dedoppler(bg, metadata, boxcar_size=2, 
                                 max_dd=4.0)

    maxpixel = np.argmax(dedopp)
    mdrift, mchan = (maxpixel // 1024, maxpixel % 1024)
    maxpixel_val = np.max(dedopp)
    print(f"dedopp recovered power (boxcar 2): {maxpixel_val}")
    assert maxpixel_val == np.sum(bg)

    # plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    imshow_waterfall(bg, metadata)
    plt.subplot(1,2,2)
    imshow_dedopp(dedopp, metadata, 'channel', 'driftrate')
    plt.savefig('docs/figs/test_dedoppler_boxcar.png')
    plt.show()
    

def test_hitsearch():
    n_timesteps = 32 
    n_chan = 4096
    signal_bw = 16

    # Create test data
    metadata = {'fch1': 1000*u.MHz, 'dt': 1.0*u.s, 'df': 1.0*u.Hz}
    frame = stg.Frame(fchans=n_chan*u.pixel, tchans=n_timesteps*u.pixel,
              df=metadata['df'], dt=metadata['dt'], fch1=metadata['fch1'])
    d = frame.add_noise(x_mean=0, x_std=1, noise_type='gaussian')
    # Add a signal with bandwidth into the data, SNR of 1000
    for ii in range(signal_bw):
        d[:, n_chan // 2 + ii]   = 1000 / signal_bw
    
    print("--- Run dedoppler() then hitsearch() ---")
    dedopp, metadata = dedoppler(d, metadata, boxcar_size=16, max_dd=1.0)
    hits0 = hitsearch(dedopp, metadata, threshold=1000).sort_values('snr')
    print(hits0)
    # Output should be 
    #driftrate      f_start      snr  driftrate_idx  channel_idx  boxcar_size
    #0        0.0  1000.002056  32000.0             32         2056           16
    # Note that SNR here is unnormalized, so actually peak value -- as we didn't renormalize
    h0 = hits0.iloc[0]
    assert h0['snr'] == 32000.0
    assert h0['channel_idx'] == 2056
    assert h0['driftrate_idx'] == 32
    assert len(hits0) == 1
    
    print("--- run_pipeline with w/o merge --- ")
    dedopp, metadata, hits = run_pipeline(d, metadata, max_dd=1.0, min_dd=None, threshold=100, 
                                          n_boxcar=7, merge_boxcar_trials=False)

    for rid, hit in hits.iterrows():
         assert(np.abs(hit['channel_idx'] - 2048) < np.max((signal_bw, hit['boxcar_size'])))
    
    print(hits)
    
    print("--- run merge_hits --- ")
    print(hits.dtypes)
    merged_hits = merge_hits(hits)
    assert len(merged_hits == 1)
    print(merged_hits)
    
    print("--- run_pipeline with merge --- ")
    dedopp, md, hits2 = run_pipeline(d, metadata, max_dd=1.0, min_dd=None, threshold=100, 
                                           n_boxcar=7, merge_boxcar_trials=True)
    hits2
    print(hits2)    
    assert hits2.iloc[0]['boxcar_size'] == signal_bw
    assert len(hits2) == len(merged_hits) == 1

    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    imshow_waterfall(d, metadata, 'channel', 'timestep')
    
    plt.subplot(1,2,2)
    imshow_dedopp(dedopp, metadata, 'channel', 'driftrate')
    
    plt.savefig('docs/figs/test_hitsearch.png')
    plt.show()

def test_hitsearch_multi():
    metadata = {'fch1': 6095.214842353016*u.MHz, 
            'dt': 18.25361108*u.s, 
            'df': 2.7939677238464355*u.Hz}

    frame = stg.Frame(fchans=2**12*u.pixel,
                      tchans=32*u.pixel,
                      df=metadata['df'],
                      dt=metadata['dt'],
                      fch1=metadata['fch1'])

    test_tones = [
      {'f_start': frame.get_frequency(index=500), 'drift_rate': 0.50*u.Hz/u.s, 'snr': 100, 'width': 20*u.Hz},
      {'f_start': frame.get_frequency(index=800), 'drift_rate': -0.40*u.Hz/u.s, 'snr': 100, 'width': 20*u.Hz},
      {'f_start': frame.get_frequency(index=2048), 'drift_rate': 0.00*u.Hz/u.s, 'snr': 20, 'width': 6*u.Hz},
      {'f_start': frame.get_frequency(index=3000), 'drift_rate': 0.07*u.Hz/u.s, 'snr': 50, 'width': 3*u.Hz}
    ]

    noise = frame.add_noise(x_mean=0, x_std=5, noise_type='gaussian')

    for tone in test_tones:
        signal = frame.add_signal(stg.constant_path(f_start=tone['f_start'],
                                                drift_rate=tone['drift_rate']),
                              stg.constant_t_profile(level=frame.get_intensity(snr=tone['snr'])),
                              stg.gaussian_f_profile(width=tone['width']),
                              stg.constant_bp_profile(level=1))
    
    d = frame.data
    
    fig = plt.figure(figsize=(10, 6))

    dedopp, md, hits = run_pipeline(d, metadata, max_dd=1.0, min_dd=None, threshold=100, 
                                    n_boxcar=5, merge_boxcar_trials=True)
    print(hits.sort_values('snr', ascending=False))
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    imshow_waterfall(d, md, 'channel', 'timestep')
    
    plt.subplot(1,2,2)
    imshow_dedopp(dedopp, md, 'channel', 'driftrate')
    overlay_hits(hits, 'channel', 'driftrate')
    
    plt.savefig('docs/figs/test_hitsearch_multi.png')
    plt.show()