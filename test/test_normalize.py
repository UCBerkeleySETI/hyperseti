from hyperseti import normalize
from hyperseti.io import from_setigen

import cupy as cp
from astropy import units as u
import pylab as plt
import numpy as np
import os
import setigen as stg


def test_normalize_basic():
    """ Basic test of normalization """
    n_int, n_ifs, n_chan = 128, 1, 8192
    np.random.seed(1234)
    d = np.random.normal(size=n_chan*n_int*n_ifs, loc=10, scale=17).astype('float32')
    d = d.reshape([n_int, n_ifs, n_chan])**2

    d_norm = normalize(d, return_space='cpu')
    print(d_norm.mean(), d_norm.std())
    assert np.isclose(d_norm.mean(), 0, atol=1e-6)
    assert np.isclose(d_norm.std(), 1.0, atol=1e-6)

def test_normalize_bandpass(plot=False):
    """ Test normalization works with polynomial bandpass included """
    n_int, n_ifs, n_chan = 128, 1, 8192
    np.random.seed(1234)
    d = np.random.normal(size=n_chan*n_int*n_ifs, loc=10, scale=17).astype('float32')
    d = d.reshape([n_int, n_ifs, n_chan])**2
    
    x = np.arange(n_chan) - n_chan//2
    d[:, 0] += -1.2e-4*x**2 + 2500
    if plot:
        plt.plot(d.mean(axis=0).squeeze())

    d_norm = normalize(d, poly_fit=2, return_space='cpu')
    print(d_norm.mean(), d_norm.std())
    if plot:
        plt.plot(d_norm.mean(axis=0).squeeze())
    assert np.isclose(d_norm.mean(), 0, atol=1e-6)
    assert np.isclose(d_norm.std(), 1.0, atol=1e-6)
    
def test_normalize_multi_if(plot=False):
    """ Test normalization on multiple IFs, each with bandpasses"""
    n_int, n_ifs, n_chan = 128, 4, 8192
    np.random.seed(1234)
    d = np.random.normal(size=n_chan*n_int*n_ifs, loc=10, scale=17).astype('float32')
    d = d.reshape([n_int, n_ifs, n_chan])**2
    x = np.arange(n_chan) - n_chan//2
    d[:, 0] += -1.2e-4*x**2 + 2500
    d[:, 1] += -1.1e-4*x**2 + 2200
    d[:, 2] += -1.25e-4*x**2 + 2700
    d[:, 3] += -1.27e-4*x**2 + 2800
    
    if plot:
        plt.subplot(2,1,1)
        plt.plot(d[:, 0].mean(axis=0))
        plt.plot(d[:, 1].mean(axis=0))
        plt.plot(d[:, 2].mean(axis=0))
        plt.plot(d[:, 2].mean(axis=0))

    d_norm = normalize(d, poly_fit=2, return_space='cpu')
    print(d_norm.mean(), d_norm.std())
    if plot:
        plt.subplot(2,1,2)
        plt.plot(d_norm[:, 0].mean(axis=0))
        plt.plot(d_norm[:, 1].mean(axis=0))
        plt.plot(d_norm[:, 2].mean(axis=0))
        plt.plot(d_norm[:, 3].mean(axis=0))
        
    assert np.isclose(d_norm.mean(), 0, atol=1e-6)
    assert np.isclose(d_norm.std(), 1.0, atol=1e-6)

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
    d.data = d.data.astype('float32')
    if plot:
        frame.plot()
    
    # Over 20% of data will be masked
    mask = d.data.mean(axis=0).squeeze() > 15
    d_norm = normalize(d, mask=mask, return_space='cpu')
    #print(d_norm.mean(), d_norm.std())
    
    plt.figure()
    if plot:
        plt.plot(d_norm.mean(axis=0).squeeze())
    
    mask_cpu = cp.asnumpy(mask)
    assert np.isclose(d_norm[..., ~mask_cpu].mean(), 0, atol=1e-6)
    assert np.isclose(d_norm[..., ~mask_cpu].std(), 1.0, atol=1e-6)
    print("Mask test passed!")


if __name__== "__main__":
    test_normalize_basic()
    test_normalize_bandpass()
    test_normalize_multi_if()
    test_normalize_mask()


