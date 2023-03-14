from hyperseti.dedoppler import dedoppler
from hyperseti.filter import apply_boxcar
from hyperseti.normalize import normalize
from hyperseti.hits import hitsearch, merge_hits
from hyperseti.io import from_fil, from_metadata
from hyperseti.log import set_log_level
from hyperseti.data_array import split_metadata
import cupy as cp

from astropy import units as u
from astropy.time import Time
from datetime import datetime

import setigen as stg
import pylab as plt
import numpy as np

from hyperseti.plotting import imshow_dedopp, imshow_waterfall, overlay_hits
try:
    from .file_defs import synthetic_fil, test_fig_dir, voyager_h5
except:
    from file_defs import synthetic_fil, test_fig_dir, voyager_h5
import os

import logbook
import hyperseti

#hyperseti.dedoppler.logger.level = logbook.DEBUG
#hyperseti.utils.logger.level = logbook.DEBUG

def test_dedoppler():
    """ Basic tests of the dedoppler functionality """

    # zero drift test, no normalization
    test_data = np.ones(shape=(32, 1, 1024), dtype='float32')
    test_data[:, :, 511] = 10
    
    metadata_in = {'frequency_start': 1000*u.MHz,
                'time_start': Time(datetime.now()),
                'time_step': 1.0*u.s, 
                'frequency_step': 1.0*u.Hz,
                'dims': ('time', 'beam_id', 'frequency')}

    dedopp_array = from_metadata(cp.asarray(test_data), metadata_in)
    
    dedopp_array = dedoppler(dedopp_array, boxcar_size=1, max_dd=1.0)
    print("type(dedopp):", type(dedopp_array))
    print("dedopp.data:", dedopp_array.data)
    print("np.max(dedopp.data):", np.max(dedopp_array.data), ", np.sum(test_data[:, :, 511]):", np.sum(test_data[:, :, 511]))
    
    #TODO: Recalculate
    #assert np.max(dedopp.data) == np.sum(test_data[:, :, 511])
    

    for dr_test in (0.0, 0.1, 0.5, -0.25, -0.5):
        # single drifting tone
        frame = stg.Frame(
                  fchans=2**10*u.pixel, tchans=32*u.pixel,
                  df=metadata_in['frequency_step'], 
                  dt=metadata_in['time_step'], 
                  fch1=metadata_in['frequency_start'])
    
        tone = {'f_start': frame.get_frequency(index=500), 
                'drift_rate': dr_test * u.Hz / u.s, 
                'snr': 500, 
                'width': metadata_in['frequency_step']}
        frame.add_noise(x_mean=1, noise_type='chi2')

        frame.add_signal(stg.constant_path(f_start=tone['f_start'],
                                                    drift_rate=tone['drift_rate']),
                                  stg.constant_t_profile(level=frame.get_intensity(snr=tone['snr'])),
                                  stg.gaussian_f_profile(width=tone['width']),
                                  stg.constant_bp_profile(level=1))

        frame.save_fil(filename=synthetic_fil)
        darray = from_fil(synthetic_fil)
        darray.data = cp.asarray(darray.data[:])

        dedopp_array = dedoppler(darray, boxcar_size=1, max_dd=1.0, plan='optimal')

        dedopp, metadata = split_metadata(dedopp_array)

        # Manual dedoppler search -- just find max channel (only works if S/N is good)
        manual_dd_tot = 0
        for ii in range(darray.data.shape[0]):
            manual_dd_tot += np.max(cp.asnumpy(darray.data[ii]))
        plt.clf()
        imshow_dedopp(dedopp_array, show_colorbar=False)
        plt.colorbar()
        plt.savefig(os.path.join(test_fig_dir, f'test_dedoppler_{dr_test}.png'))

        maxpixel = np.argmax(dedopp_array.data)
        mdrift, mchan = (maxpixel // 1024, maxpixel % 1024)
        drates = cp.asnumpy(dedopp_array.drift_rate.data)
        optimal_drift = drates[int(mdrift)]
        maxpixel_val = np.max(cp.asnumpy(dedopp_array.data))
        
        frac_recovered = (maxpixel_val / manual_dd_tot)

        print(f"Inserted drift rate:  {tone['drift_rate']} \tSUM: {manual_dd_tot:2.2f}")
        print(f"Recovered drift rate: {optimal_drift} Hz / s \tSUM: {maxpixel_val:2.2f}\n")

        # Channel should detected at +/- 1 chan
        print(mdrift, mchan)
        #assert np.abs(mchan - 500) <= 1
        
        # Drift rate should be detected +/- 1 drift resolution
        drift_rate_step = 0.03125
        assert np.abs(optimal_drift - dr_test) <= 1.01*np.abs(drift_rate_step)

        # TODO: Fix stats for this
        # Recovered signal sum should be close to manual method
        #assert 1.001 >= frac_recovered >= 0.825

    # Finish off figure plotting
    plt.show()

def test_dedoppler_boxcar():
    """ Test that boxcar averaging works as expected """
    def generate_drifting_tone(n_chan, n_timesteps, n_drift_per_step, n_beams=1, sigval=10):
        """ Simple tone generator to generate smeared tones """
        bg = np.zeros((n_timesteps, n_beams, n_chan), dtype='float32')

        for ii in range(0, bg.shape[0]):
            for nd in range(n_drift_per_step):
                z = n_drift_per_step * ii + nd
                bg[ii, :, bg.shape[2]//2 + z] = sigval / n_drift_per_step
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
    metadata_in = {'frequency_start': 1000*u.MHz,
                'time_start': Time(datetime.now()),
                'time_step': 1.0*u.s, 
                'frequency_step': 1.0*u.Hz,
                'dims': ('time', 'beam_id', 'frequency')}

    bg = generate_drifting_tone(n_chan=256, n_timesteps=32, n_drift_per_step=2, sigval=10)
    print(f"Total power in frame: {np.sum(bg)}")

    # Compute dedoppler using basic maxhold method
    maxhold_res = maxhold_dedoppler(bg)
    print(f"MAXHOLD recovered power: {maxhold_res}")

    # With boxcar_size = 1 we should recover 160
    dedopp_array = dedoppler(from_metadata(cp.asarray(bg), metadata_in), boxcar_size=1,
                                 max_dd=2.0, plan='optimal')

    maxpixel = np.argmax(cp.asnumpy(dedopp_array.data))
    mdrift, mchan = (maxpixel // 1024, maxpixel % 1024)
    maxpixel_val = np.max(cp.asnumpy(dedopp_array.data))
    print(f"dedopp recovered power (boxcar 1): {maxpixel_val}")
    
    #TODO: FIX assertion
    #assert maxpixel_val == maxhold_res

    # With boxcar_size = 2 we should recover 320 (full amount)
    bg = generate_drifting_tone(n_chan=256, n_timesteps=32, n_drift_per_step=2, sigval=10)
    dedopp = dedoppler(from_metadata(cp.asarray(bg), metadata_in), boxcar_size=2, 
                                 max_dd=4.0, plan='optimal')

    maxpixel = np.argmax(cp.asnumpy(dedopp.data))
    mdrift, mchan = (maxpixel // 1024, maxpixel % 1024) # <----------- UNUSED
    maxpixel_val = np.max(cp.asnumpy(dedopp.data))
    print(f"dedopp recovered power (boxcar 2): {maxpixel_val}")
    
    #TODO: Fix assertion
    #assert maxpixel_val == np.sum(bg)

    # plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    imshow_waterfall(from_metadata(bg, metadata_in))
    plt.subplot(1,2,2)
    imshow_dedopp(dedopp, 'channel', 'driftrate')
    plt.savefig(os.path.join(test_fig_dir, 'test_dedoppler_boxcar.png'))
    plt.show()
    
if __name__ == "__main__":
    test_dedoppler()
    test_dedoppler_boxcar()
    