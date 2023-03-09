import cupy as cp
import numpy as np
import time
import os

from .utils import on_gpu, datwrapper

#logging
from .log import get_logger
logger = get_logger('hyperseti.normalize')


def normalize(data_array,  mask=None, poly_fit=0):
    """ Apply normalization on GPU
    
    Applies normalisation (data - mean) / stdev
    
    Args: 
        data (np/cp.array): Data to preprocess
        mask (np.cp.array): 1D Channel mask for RFI flagging
        return_space ('cpu' or 'gpu'): Returns array in CPU or GPU space
        poly_fit (int): Fit polynomial of degree N, 0 = no fit.
        
    Returns: d_gpu (cp.array): Normalized data
    """
    # Normalise
    logger.debug(f"Poly fit = {poly_fit}")
    t0 = time.time()

    d_flag = cp.copy(data_array.data)

    n_int, n_ifs, n_chan = data_array.data.shape

    # Setup 1D channel mask -- used for polynomial fitting
    if mask is None: 
        mask = cp.zeros(n_chan, dtype='bool')

    # Do polynomial fit and compute stats (with masking)
    d_mean_ifs, d_std_ifs = cp.zeros(n_ifs), cp.zeros(n_ifs)

    N_masked = mask.sum()
    N_flagged = N_masked * n_ifs * n_int
    N_tot     = np.product(data_array.data.shape)
    N_unflagged = (N_tot - N_flagged)

    t0p = time.time()
    
    for ii in range(n_ifs):
        x    = cp.arange(n_chan, dtype='float64') 
        xc   = cp.compress(~mask, x)
        dfit = cp.compress(~mask, data_array.data[:, ii].mean(axis=0))

        if poly_fit > 0:
            # WAR: int64 dtype causes issues in cupy 10 (19.04.2022)
            p    = cp.poly1d(cp.polyfit(xc, dfit, poly_fit))
            fit   = p(x)
            dfit  -=  p(xc)
            data_array.data[:, ii] = data_array.data[:, ii] - fit

        # compute mean and stdev
        dmean = dfit.mean()
        dvar  = ((data_array.data[:, ii] - dmean)**2).mean(axis=0)
        dvar  = cp.compress(~mask, dvar).mean()
        dstd  = cp.sqrt(dvar)
        d_mean_ifs[ii] = dmean
        d_std_ifs[ii]  = dstd

    t1p = time.time()
    logger.debug(f"Mean+Std time: {(t1p-t0p)*1e3:2.2f}ms")

    flag_fraction =  N_flagged / N_tot
    ## flag_correction =  N_tot / (N_tot - N_flagged) # <---------------------- unused
    logger.debug(f"Flagged fraction: {flag_fraction:2.4f}")
    if flag_fraction > 0.2:
        logger.warning(f"High flagged fraction: {flag_fraction:2.3f}")

    #  Apply to original data
    for ii in range(n_ifs):
        data_array.data[:, ii] = ((data_array.data[:, ii] - d_mean_ifs[ii]) / d_std_ifs[ii])
    t1 = time.time()
    logger.debug(f"Normalisation time: {(t1-t0)*1e3:2.2f}ms")
    
    return data_array
