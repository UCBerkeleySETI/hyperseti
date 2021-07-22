import cupy as cp
import numpy as np
import time
import os

from astropy import units as u
from cupyx.scipy.ndimage import uniform_filter1d

from .utils import on_gpu, datwrapper
from .gpu_kernels import dedoppler_kernel, dedoppler_kurtosis_kernel

#logging
from .log import logger_group, Logger
logger = Logger('hyperseti.dedoppler')
logger_group.add_logger(logger)


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
    logger.info(f"Running boxcar mode {mode} with size {boxcar_size}")
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
    logger.info(f"Filter time: {(t1-t0)*1e3:2.2f}ms")
    
    return data

  
@datwrapper(dims=('drift_rate', 'beam_id', 'frequency'))
@on_gpu  
def dedoppler(data, metadata, max_dd, min_dd=None, boxcar_size=1, beam_id=0,
              boxcar_mode='sum', kernel='dedoppler'):
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
    
    metadata['drift_rates'] = dd_vals * u.Hz / u.s
    metadata['drift_rate_start'] = dd_vals[0] * u.Hz / u.s
    metadata['drift_rate_step']  = delta_dd * u.Hz / u.s
    metadata['boxcar_size'] = boxcar_size
    # metadata['time_step'] = obs_len * u.s   # NB: Why did I do this?
    dedopp_gpu = cp.expand_dims(dedopp_gpu, axis=1)
    return dedopp_gpu, metadata