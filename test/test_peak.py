
import os
import logbook
from hyperseti import normalize, dedoppler
from hyperseti.io import from_setigen
from hyperseti.log import set_log_level
from hyperseti.peak import find_peaks_argrelmax
import numpy as np
import setigen as stg
from astropy import units as u

def test_argrelmax():
    """ Test argrelmax works """
    set_log_level("info")
    np.random.seed(1234)

    metadata = {'fch1': 6095.214842353016*u.MHz, 
                'dt': 18.25361108*u.s, 
                'df': 2.7939677238464355*u.Hz}

    frame = stg.Frame(fchans=2**12*u.pixel,
                    tchans=32*u.pixel,
                    df=metadata['df'],
                    dt=metadata['dt'],
                    fch1=metadata['fch1'])

    test_tones = [
    {'f_start': frame.get_frequency(index=500), 'drift_rate': 0.70*u.Hz/u.s, 'snr': 100, 'width': 20*u.Hz},
    {'f_start': frame.get_frequency(index=700), 'drift_rate': -0.55*u.Hz/u.s, 'snr': 100, 'width': 20*u.Hz},
    {'f_start': frame.get_frequency(index=2048), 'drift_rate': 0.00*u.Hz/u.s, 'snr': 40, 'width': 6*u.Hz},
    {'f_start': frame.get_frequency(index=3000), 'drift_rate': 0.07*u.Hz/u.s, 'snr': 50, 'width': 3*u.Hz}
    ]

    noise = frame.add_noise(x_mean=10, x_std=5, noise_type='chi2')

    for tone in test_tones:
        signal = frame.add_signal(stg.constant_path(f_start=tone['f_start'],
                                                drift_rate=tone['drift_rate']),
                            stg.constant_t_profile(level=frame.get_intensity(snr=tone['snr'])),
                            stg.gaussian_f_profile(width=tone['width']),
                            stg.constant_bp_profile(level=1))

    d_arr = from_setigen(frame)
    d_arr.data = normalize(d_arr)

    dd, md = dedoppler(d_arr, max_dd=1.0, apply_smearing_corr=False)
    print(type(dd))

    snrs, fidx, ddidx =  find_peaks_argrelmax(dd, threshold=20, order=100)
    assert len(snrs) == 4
 
    assert np.allclose(snrs, [74.470345, 74.89874,  28.147993, 25.853714])
    assert np.allclose(fidx, [ 500,  700, 2048, 3001])
    assert np.allclose(ddidx, [356,  93, 208, 223])    

if __name__ == "__main__":
    dframe = test_argrelmax()
