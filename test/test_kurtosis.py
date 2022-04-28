from hyperseti.io import from_fil, from_h5
from hyperseti.kurtosis import spectral_kurtosis, sk_flag
from .file_defs import synthetic_fil, test_fig_dir, voyager_h5

import cupy as cp
from astropy import units as u
import pylab as plt
import numpy as np
import os

import hyperseti

def test_kurtosis():
    metadata = {'frequency_start': 8421.38671875*u.MHz, 
                'time_step': 18.253611007999982*u.s, 
                'frequency_step': -2.7939677238464355e-06*u.MHz}
    h5 = from_h5(voyager_h5)

    d = spectral_kurtosis(h5, metadata)
    print("after spectral_kurtosis, d.shape: {}, d: {}".format(d.shape, d))
    
    m = sk_flag(h5, metadata, n_sigma_upper=3, n_sigma_lower=2, flag_upper=True, flag_lower=True)
    print("Mask after sk_flag:", m)
    m = sk_flag(h5, metadata, n_sigma_upper=3, n_sigma_lower=2, flag_upper=False, flag_lower=True)
    print("Mask after sk_flag:", m)
    m = sk_flag(h5, metadata, n_sigma_upper=3, n_sigma_lower=2, flag_upper=True, flag_lower=False)
    print("Mask after sk_flag:", m)

if __name__ == "__main__":
    test_kurtosis()
