from hyperseti.data_array import from_fil, from_h5
from hyperseti.kurtosis import spectral_kurtosis, sk_flag
from .file_defs import synthetic_fil, test_fig_dir, voyager_h5

import cupy as cp
from astropy import units as u
import pylab as plt
import numpy as np
import os

import logbook
import hyperseti
#hyperseti.dedoppler.logger.level = logbook.DEBUG
#hyperseti.utils.logger.level = logbook.DEBUG

def test_kurtosis():
    h5 = from_h5(voyager_h5)
    d = spectral_kurtosis(h5)
    
    m = sk_flag(h5, n_sigma_upper=3, n_sigma_lower=2, flag_upper=True, flag_lower=True)
    m = sk_flag(h5, n_sigma_upper=3, n_sigma_lower=2, flag_upper=False, flag_lower=True)
    m = sk_flag(h5, n_sigma_upper=3, n_sigma_lower=2, flag_upper=True, flag_lower=False)
    
if __name__ == "__main__":
    test_kurtosis()
