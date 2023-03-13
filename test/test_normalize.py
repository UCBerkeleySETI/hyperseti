from hyperseti.normalize import normalize
from hyperseti.io import from_setigen
from hyperseti.data_array import from_metadata

import cupy as cp
from astropy import units as u
import pylab as plt
import numpy as np
import os
import setigen as stg
from astropy.time import Time 


def create_data_array(data):
    metadata_in = {'frequency_start': 1000*u.MHz,
                'time_start': Time(60000, format='mjd'),
                'time_step': 1.0*u.s, 
                'frequency_step': 1.0*u.Hz,
                'dims': ('time', 'beam_id', 'frequency')}
    return from_metadata(cp.asarray(data), metadata_in)

def test_normalize_basic():
    """ Basic test of normalization """
    n_int, n_ifs, n_chan = 128, 1, 8192
    np.random.seed(1234)
    d = np.random.normal(size=n_chan*n_int*n_ifs, loc=10, scale=17).astype('float32')
    d = d.reshape([n_int, n_ifs, n_chan])**2
    d = create_data_array(d)

    d_norm = normalize(d)
    print(d_norm.data.mean(), d_norm.data.std())
    assert np.isclose(d_norm.data.mean(), 0, atol=1e-6)
    assert np.isclose(d_norm.data.std(), 1.0, atol=1e-6)

def test_normalize_bandpass(plot=False):
    """ Test normalization works with polynomial bandpass included """
    n_int, n_ifs, n_chan = 128, 1, 8192
    np.random.seed(1234)
    d = np.random.normal(size=n_chan*n_int*n_ifs, loc=10, scale=17).astype('float32')
    d = d.reshape([n_int, n_ifs, n_chan])**2
    d = create_data_array(d)

    x = cp.arange(n_chan) - n_chan//2
    d.data[:, 0] += -1.2e-4*x**2 + 2500
    if plot:
        plt.plot(d.data.mean(axis=0).squeeze())

    d_norm = normalize(d, poly_fit=2)
    print(d_norm.data.mean(), d_norm.data.std())
    if plot:
        plt.plot(d_norm.data.mean(axis=0).squeeze())
    assert np.isclose(d_norm.data.mean(), 0, atol=1e-6)
    assert np.isclose(d_norm.data.std(), 1.0, atol=1e-6)
    
def test_normalize_multi_if(plot=False):
    """ Test normalization on multiple IFs, each with bandpasses"""
    n_int, n_ifs, n_chan = 128, 4, 8192
    np.random.seed(1234)
    d = np.random.normal(size=n_chan*n_int*n_ifs, loc=10, scale=17).astype('float32')
    d = d.reshape([n_int, n_ifs, n_chan])**2
    d = create_data_array(d)

    x = cp.arange(n_chan) - n_chan//2
    d.data[:, 0] += -1.2e-4*x**2 + 2500
    d.data[:, 1] += -1.1e-4*x**2 + 2200
    d.data[:, 2] += -1.25e-4*x**2 + 2700
    d.data[:, 3] += -1.27e-4*x**2 + 2800
    
    if plot:
        plt.subplot(2,1,1)
        plt.plot(d.data[:, 0].mean(axis=0))
        plt.plot(d.data[:, 1].mean(axis=0))
        plt.plot(d.data[:, 2].mean(axis=0))
        plt.plot(d.data[:, 2].mean(axis=0))

    d_norm = normalize(d, poly_fit=2)
    d.data, d_norm.data = cp.asnumpy(d.data), cp.asnumpy(d_norm.data)

    print(d_norm.data.mean(), d_norm.data.std())
    if plot:
        plt.subplot(2,1,2)
        plt.plot(d_norm.data[:, 0].mean(axis=0))
        plt.plot(d_norm.data[:, 1].mean(axis=0))
        plt.plot(d_norm.data[:, 2].mean(axis=0))
        plt.plot(d_norm.data[:, 3].mean(axis=0))
        
    assert np.isclose(d_norm.data.mean(), 0, atol=1e-6)
    assert np.isclose(d_norm.data.std(), 1.0, atol=1e-6)

def test_normalize_mask(plot=False):
    """ Test normalization works when data are masked """
    n_int, n_ifs, n_chan = 16, 1, 8192
    np.random.seed(1234)
    
    frame = stg.Frame(fchans=1024*u.pixel,
                      tchans=32*u.pixel,
                      df=2.7939677238464355*u.Hz,
                      dt=18.253611008*u.s,
                      fch1=6095.214842353016*u.MHz)
    noise = frame.add_noise(x_mean=10, noise_type='chi2')
    signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(index=200),
                                                drift_rate=1*u.Hz/u.s),
                              stg.constant_t_profile(level=frame.get_intensity(snr=1000)),
                              stg.gaussian_f_profile(width=20*u.Hz),
                              stg.constant_bp_profile(level=1))
    d = from_setigen(frame)
    d.data = cp.asarray(d.data.astype('float32'))
    if plot:
        frame.plot()
    
    # Over 20% of data will be masked
    mask = d.data.mean(axis=0).squeeze() > 15
    d_norm = normalize(d, mask=mask)
    #print(d_norm.mean(), d_norm.std())
    
    plt.figure()
    if plot:
        plt.plot(d_norm.mean(axis=0).squeeze())
    
    mask_cpu = cp.asnumpy(mask)
    d_norm.data = cp.asnumpy(d_norm.data)
    
    assert np.isclose(d_norm.data[..., ~mask_cpu].mean(), 0, atol=1e-6)
    assert np.isclose(d_norm.data[..., ~mask_cpu].std(), 1.0, atol=1e-6)
    print("Mask test passed!")


if __name__== "__main__":
    test_normalize_basic()
    test_normalize_bandpass()
    test_normalize_multi_if()
    test_normalize_mask()


