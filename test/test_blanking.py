from hyperseti.blanking import blank_edges, blank_extrema, blank_hits
from hyperseti.io import from_setigen
import setigen as stg
from astropy import units as u
import cupy as cp
import numpy as np

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

if __name__ == "__main__":
    test_blank_edges()
    test_blank_extrema()
