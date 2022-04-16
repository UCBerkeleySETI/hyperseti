import cupy as cp
import numpy as np
import time
import os

from .utils import on_gpu, datwrapper

#logging
from .log import get_logger
logger = get_logger('hyperseti.normalize')


@on_gpu
def normalize(data,  mask=None, padding=0):
    """ Apply normalization on GPU
    
    Applies normalisation (data - mean) / stdev
    
    Args: 
        data (np/cp.array): Data to preprocess
        mask (np.cp.array): 1D Channel mask for RFI flagging
        padding (int): size of edge region to discard (e.g. coarse channel edges)
        return_space ('cpu' or 'gpu'): Returns array in CPU or GPU space
        
    Returns: d_gpu (cp.array): Normalized data
    """

    d_gpu = data
    d_gpu_flagged = cp.asarray(data.astype('float32', copy=True))
    
    paddingu = None if padding == 0 else -padding
    
    # Need to correct stats 
    N_flagged = 0 
    N_tot     = np.product(d_gpu[..., padding:paddingu].shape)
    if mask is not None:
        # Convert 1D-mask to match data dimensions
        mask_gpu = cp.repeat(cp.asarray(mask.reshape((1, 1, len(mask)))), d_gpu.shape[0], axis=0)
        cp.putmask(d_gpu_flagged, mask_gpu, 0)
        N_flagged = mask_gpu[..., padding:paddingu].sum()
        
    # Normalise
    t0 = time.time()
    # Compute stats based off flagged arrays
    d_mean = cp.mean(d_gpu_flagged[..., padding:paddingu])
    d_std  = cp.std(d_gpu_flagged[..., padding:paddingu])
    flag_correction =  N_tot / (N_tot - N_flagged) 
    d_mean = d_mean * flag_correction
    d_std  = d_std * np.sqrt(flag_correction)
    logger.debug(f"flag fraction correction factor: {flag_correction}")
    
    #  Apply to original data
    d_gpu = (d_gpu - d_mean) / d_std
    t1 = time.time()
    logger.info(f"Normalisation time: {(t1-t0)*1e3:2.2f}ms")
    
    return d_gpu