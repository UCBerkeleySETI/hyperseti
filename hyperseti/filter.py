import cupy as cp
import numpy as np
import time
import os
from copy import deepcopy

from cupyx.scipy.ndimage import uniform_filter1d

from .utils import on_gpu, datwrapper

#logging
from .log import get_logger
logger = get_logger('hyperseti.filter')


@on_gpu
def apply_boxcar(data, boxcar_size=1, axis=1, mode='mean'):
    """ Apply moving boxcar filter and renormalise by sqrt(boxcar_size)
    
    Boxcar applies a moving MEAN to the data. 
    Optionally apply sqrt(N) factor to keep stdev of gaussian noise constant.
    
    Args:
        data (np/cp.array): Data to apply boxcar to
        boxcar_size (int): Size of boxcar filter
        mode (str): Choose one of 'mean', 'mode', 'gaussian'
                    Where gaussian multiplies by sqrt(N) to maintain
                    stdev of Gaussian noise
        return_space ('cpu' or 'gpu'): Return in CPU or GPU space
    
    Returns: 
        data (np/cp.array): Data after boxcar filtering.
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

@datwrapper(dims=('drift_rate', 'feed_id', 'frequency'))
@on_gpu
def apply_boxcar_drift(data, metadata):
    """ Apply boxcar filter to compensate for doppler smearing
    
    An optimal boxcar is applied per row of drift rate. This retrieves
    a sensitivity increase of sqrt(boxcar_size) for a smeared signal.
    (Stil down a sqrt(boxcar_size) compared to no smearing case).
    
    Args:
        data (np or cp array): 
        metadata (dict): Dictionary of metadata values
    
    Returns:
        data, metadata (array and dict): Data array with filter applied.
    """
    logger.debug(f"apply_boxcar_drift: Applying moving average based on drift rate.")
    metadata = deepcopy(metadata)
    # Compute drift rates from metadata
    dr0, ddr = metadata['drift_rate_start'].value, metadata['drift_rate_step'].value
    df = metadata['frequency_step'].to('Hz').value
    dt = metadata['integration_time'].to('s').value
    drates =  dr0  + ddr * cp.arange(data.shape[0])
    
    # Compute smearing (array of n_channels smeared for given driftrate)
    smearing_nchan = cp.abs(dt * drates / df).astype('int32')
    smearing_nchan_max = cp.asnumpy(cp.max(smearing_nchan))

    # Apply boxcar filter to compensate for smearing
    for boxcar_size in range(2, smearing_nchan_max+1):
        idxs = cp.where(smearing_nchan == boxcar_size)
        # 1. uniform_filter1d computes mean. We want sum, so *= boxcar_size
        # 2. we want noise to stay the same, so divide by sqrt(boxcar_size)
        # combined 1 and 2 give aa sqrt(2) factor
        data[idxs] = uniform_filter1d(data[idxs], size=boxcar_size, axis=2) * np.sqrt(boxcar_size)
    return data, metadata