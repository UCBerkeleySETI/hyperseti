import cupy as cp
import numpy as np
import time
import os
from copy import deepcopy

from astropy import units as u
from cupyx.scipy.ndimage import uniform_filter1d

from .utils import on_gpu, datwrapper
from .gpu_kernels import dedoppler_kernel, dedoppler_kurtosis_kernel
from .filter import apply_boxcar_drift

#logging
from .log import logger_group, Logger
logger = Logger('hyperseti.dedoppler')
logger_group.add_logger(logger)
  
@datwrapper(dims=('drift_rate', 'feed_id', 'frequency'))
@on_gpu  
def dedoppler(data, metadata, max_dd, min_dd=None, boxcar_size=1, beam_id=0,
              boxcar_mode='sum', kernel='dedoppler', apply_smearing_corr=True):
    """ Apply brute-force dedoppler kernel to data
    
    Args:
        data (np.array): Numpy array with shape (N_timestep, N_channel)
        metadata (dict): Metadata dictionary, should contain 'df' and 'dt'
                         (frequency and time resolution)
        max_dd (float): Maximum doppler drift in Hz/s to search out to.
        min_dd (float): Minimum doppler drift to search.
        boxcar_mode (str): Boxcar mode to apply. mean/sum/gaussian.
        kernel (str): 'dedoppler' or 'kurtosis'
    
    Returns:
        dd_vals, dedopp_gpu (np.array, np/cp.array): 
    """
   
    t0 = time.time()
    metadata = deepcopy(metadata)

    if min_dd is None:
        min_dd = np.abs(max_dd) * -1
    
    # Compute minimum possible drift (delta_dd)
    N_time, N_beam, N_chan = data.shape
    data = data[:, beam_id, :] # TODO ADD POL SUPPORT
        
    obs_len  = N_time * metadata['time_step'].to('s').value
    delta_dd = metadata['frequency_step'].to('Hz').value / obs_len  # e.g. 2.79 Hz / 300 s = 0.0093 Hz/s
    
    # Compute dedoppler shift schedules
    N_dopp_upper   = int(max_dd / delta_dd)
    N_dopp_lower   = int(min_dd / delta_dd)
    
    if max_dd == 0 and min_dd is None:
        dd_shifts = np.array([0], dtype='int32')
    elif N_dopp_upper > N_dopp_lower:
        dd_shifts      = np.arange(N_dopp_lower, N_dopp_upper + 1, dtype='int32')
    else:
        dd_shifts      = np.arange(N_dopp_upper, N_dopp_lower + 1, dtype='int32')[::-1]
    
    dd_shifts_gpu  = cp.asarray(dd_shifts)
    N_dopp = len(dd_shifts)
    
    # Copy data over to GPU
    d_gpu = data
    
    # Apply boxcar filter
    if boxcar_size > 1:
        d_gpu = apply_boxcar(d_gpu, boxcar_size=boxcar_size, mode='sum', return_space='gpu')
    
    # Allocate GPU memory for dedoppler data
    dedopp_gpu = cp.zeros((N_dopp, N_chan), dtype=cp.float32)
    t1 = time.time()
    logger.info(f"Dedopp setup time: {(t1-t0)*1e3:2.2f}ms")
    
    # Launch kernel
    t0 = time.time()
    
    # Setup grid and block dimensions
    F_block = np.min((N_chan, 1024))
    F_grid  = N_chan // F_block
    #print(dd_shifts)
    logger.debug(f"Kernel shape (grid, block) {(F_grid, N_dopp), (F_block,)}")
    if kernel == 'dedoppler':
        logger.debug(f"{type(d_gpu)}, {type(dedopp_gpu)}, {N_chan}, {N_time}")
        dedoppler_kernel((F_grid, N_dopp), (F_block,), 
                         (d_gpu, dedopp_gpu, dd_shifts_gpu, N_chan, N_time)) # grid, block and arguments
        
        
    elif kernel == 'kurtosis':
         # output must be scaled by N_acc, which can be figured out from df and dt metadata
        samps_per_sec = (1.0 / np.abs(metadata['frequency_step'])).to('s') / 2 # Nyq sample rate for channel
        N_acc = int(metadata['time_step'].to('s') / samps_per_sec)
        logger.debug(f'rescaling SK by {N_acc}')
        logger.debug(f"driftrates: {dd_shifts}")
        dedoppler_kurtosis_kernel((F_grid, N_dopp), (F_block,), 
                         (d_gpu, dedopp_gpu, dd_shifts_gpu, N_chan, N_time, N_acc)) # grid, block and arguments 
        
    t1 = time.time()
    logger.info(f"Dedopp kernel time: {(t1-t0)*1e3:2.2f}ms")
    
    # Compute drift rate values in Hz/s corresponding to dedopp axis=0
    dd_vals = dd_shifts * delta_dd
    
    metadata['drift_rate_start'] = dd_vals[0] * u.Hz / u.s
    metadata['drift_rate_step']  = delta_dd * u.Hz / u.s
    metadata['drift_rates'] = dd_vals
    metadata['boxcar_size'] = boxcar_size
    metadata['n_integration'] = N_time
    metadata['integration_time'] = metadata['time_step']
    metadata['obs_len'] = obs_len * u.s

    dedopp_gpu = cp.expand_dims(dedopp_gpu, axis=1)

    if apply_smearing_corr:
        logger.debug(f"Applying smearing correction")
        dedopp_darr, metadata = apply_boxcar_drift(dedopp_gpu, metadata)
        dedopp_gpu = dedopp_darr.data
    return dedopp_gpu, metadata
