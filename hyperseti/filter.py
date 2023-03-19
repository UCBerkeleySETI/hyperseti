import cupy as cp
import numpy as np
import time
import os
from copy import deepcopy

from cupyx.scipy.ndimage import uniform_filter1d

#logging
from .log import get_logger
logger = get_logger('hyperseti.filter')


def apply_boxcar(data: cp.ndarray, boxcar_size: int=1, axis: int=1, mode: str='mean') -> cp.ndarray:
    """ Apply moving boxcar filter and renormalise by sqrt(boxcar_size)
    
    Boxcar applies a moving MEAN to the data. 
    Optionally apply sqrt(N) factor to keep stdev of gaussian noise constant.
    
    Args:
        data (cp.ndarray): Data to apply boxcar to
        boxcar_size (int): Size of boxcar filter
        mode (str): Choose one of 'mean', 'mode', 'gaussian'
                    Where gaussian multiplies by sqrt(N) to maintain
                    stdev of Gaussian noise
    
    Returns: 
        data (cp.ndarray): Data after boxcar filtering.
    """
    logger.debug(f"apply_boxcar: Running boxcar mode {mode} with size {boxcar_size}")
    if mode not in ('sum', 'mean', 'gaussian'):
        raise RuntimeError("Unknown mode. Only modes sum, mean or gaussian supported.")
    t0 = time.time()
    # This keeps stdev noise the same instead of decreasing by sqrt(N)
    data = uniform_filter1d(data, size=boxcar_size, axis=axis)
    if mode == 'gaussian':
        data *= np.sqrt(boxcar_size)
    elif mode == 'sum':
        data *= boxcar_size
    t1 = time.time()
    logger.debug(f"apply_boxcar: Filter time: {(t1-t0)*1e3:2.2f}ms")
    
    return data

