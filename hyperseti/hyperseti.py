import cupy as cp
import numpy as np
import time
import pandas as pd
import logging
import os

from astropy import units as u
import setigen as stg

from cupyx.scipy.ndimage import uniform_filter1d

from .peak import prominent_peaks
from .data_array import from_fil, from_h5
from .utils import on_gpu
from .gpu_kernels import dedoppler_kernel, dedoppler_kurtosis_kernel

#logging
from .log import logger_group, Logger
logger = Logger('hyperseti.hyperseti')
logger_group.add_logger(logger)

# Max threads setup 
os.environ['NUMEXPR_MAX_THREADS'] = '8'

@on_gpu
def normalize(data, mask=None, padding=0):
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
def apply_boxcar(data, boxcar_size, axis=1, mode='mean'):
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
    if mode not in ('sum', 'mean', 'gaussian'):
        raise RuntimeError("Unknown mode. Only modes sum, mean or gaussian supported.")
    t0 = time.time()
    # Move to GPU as required, and multiply by sqrt(boxcar_size)
    # This keeps stdev noise the same instead of decreasing by sqrt(N)
    data = cp.asarray(data.astype('float32', copy=False))
    data = uniform_filter1d(data, size=boxcar_size, axis=axis)
    if mode == 'gaussian':
        data *= np.sqrt(boxcar_size)
    elif mode == 'sum':
        data *= boxcar_size
    t1 = time.time()
    logger.info(f"Filter time: {(t1-t0)*1e3:2.2f}ms")
    
    return data
    
@on_gpu    
def dedoppler(data, metadata, max_dd, min_dd=None, boxcar_size=1,
              boxcar_mode='sum', kernel='dedoppler'):
    """ Apply brute-force dedoppler kernel to data
    
    Args:
        data (np.array): Numpy array with shape (N_timestep, N_channel)
        metadata (dict): Metadata dictionary, should contain 'df' and 'dt'
                         (frequency and time resolution)
        max_dd (float): Maximum doppler drift in Hz/s to search out to.
        min_dd (float): Minimum doppler drift to search.
        boxcar_mode (str): Boxcar mode to apply. mean/sum/gaussian.
        return_space ('cpu' or 'gpu'): Returns array in CPU or GPU space
    
    Returns:
        dd_vals, dedopp_gpu (np.array, np/cp.array): 
    """
    t0 = time.time()
    if min_dd is None:
        min_dd = np.abs(max_dd) * -1
    
    # Compute minimum possible drift (delta_dd)
    N_time, N_beam, N_chan = data.shape
    if N_beam == 1:
        data = data.squeeze()
    else:
        data = data[:, 0, :] # TODO ADD POL SUPPORT
        
    obs_len  = N_time * metadata['dt'].to('s').value
    delta_dd = metadata['df'].to('Hz').value / obs_len  # e.g. 2.79 Hz / 300 s = 0.0093 Hz/s
    
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
    d_gpu = cp.asarray(data.astype('float32', copy=False))
    
    # Apply boxcar filter
    if boxcar_size > 1:
        d_gpu = apply_boxcar(d_gpu, boxcar_size, mode='sum', return_space='gpu')
    
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
        dedoppler_kernel((F_grid, N_dopp), (F_block,), 
                         (d_gpu, dedopp_gpu, dd_shifts_gpu, N_chan, N_time)) # grid, block and arguments
        
    elif kernel == 'kurtosis':
         # output must be scaled by N_acc, which can be figured out from df and dt metadata
        samps_per_sec = (1.0 / np.abs(metadata['df'])).to('s') / 2 # Nyq sample rate for channel
        N_acc = int(metadata['dt'].to('s') / samps_per_sec)
        logger.debug(f'rescaling SK by {N_acc}')
        logger.debug(f"driftrates: {dd_shifts}")
        dedoppler_kurtosis_kernel((F_grid, N_dopp), (F_block,), 
                         (d_gpu, dedopp_gpu, dd_shifts_gpu, N_chan, N_time, N_acc)) # grid, block and arguments 
        
    t1 = time.time()
    logger.info(f"Dedopp kernel time: {(t1-t0)*1e3:2.2f}ms")
    
    # Compute drift rate values in Hz/s corresponding to dedopp axis=0
    dd_vals = dd_shifts * delta_dd
    
    metadata['drift_trials'] = dd_vals
    metadata['boxcar_size'] = boxcar_size
    metadata['dd'] = delta_dd * u.Hz / u.s
    
    return dedopp_gpu, metadata


    
def create_empty_hits_table():
    """ Create empty pandas dataframe for hit data
    
    Notes:
        Columns are:
            Driftrate (float64): Drift rate in Hz/s
            f_start (float64): Frequency in MHz at start time
            snr (float64): Signal to noise ratio for detection.
            driftrate_idx (int): Index of array corresponding to driftrate
            channel_idx (int): Index of frequency channel for f_start
            boxcar_size (int): Size of boxcar applied to data
    
    Returns:
        hits (pd.DataFrame): Data frame with columns as above.
    """
    # Create empty dataframe
    hits = pd.DataFrame({'driftrate': pd.Series([], dtype='float64'),
                          'f_start': pd.Series([], dtype='float64'),
                          'snr': pd.Series([], dtype='float64'),
                          'driftrate_idx': pd.Series([], dtype='int'),
                          'channel_idx': pd.Series([], dtype='int'),
                          'boxcar_size': pd.Series([], dtype='int'),
                         })
    return hits


def hitsearch(dedopp, metadata, threshold=10, min_fdistance=None, min_ddistance=None):
    """ Search for hits using _prominent_peaks method in cupyimg.skimage
    
    Args:
        dedopp (np.array): Dedoppler search array of shape (N_trial, N_chan)
        drift_trials (np.array): List of dedoppler trials corresponding to dedopp N_trial axis
        metadata (dict): Dictionary of metadata needed to convert from indexes to frequencies etc
        threshold (float): Threshold value (absolute) above which a peak can be considered
        min_fdistance (int): Minimum distance in pixels to nearest peak along frequency axis
        min_ddistance (int): Minimum distance in pixels to nearest peak along doppler axis
    
    Returns:
        results (pd.DataFrame): Pandas dataframe of results, with columns 
                                    driftrate: Drift rate in hz/s
                                    f_start: Start frequency channel
                                    snr: signal to noise ratio
                                    driftrate_idx: Index in driftrate array
                                    channel_idx: Index in frequency array
    """
    
    drift_trials = metadata['drift_trials']
    
    if min_fdistance is None:
        min_fdistance = metadata['boxcar_size'] * 2
    
    if min_ddistance is None:
        min_ddistance = len(drift_trials) // 4

    # Copy over to GPU if required
    dedopp_gpu = cp.asarray(dedopp.astype('float32', copy=False))
    
    t0 = time.time()
    intensity, fcoords, dcoords = prominent_peaks(dedopp_gpu, min_xdistance=min_fdistance, min_ydistance=min_ddistance, threshold=threshold)
    t1 = time.time()
    logger.info(f"Peak find time: {(t1-t0)*1e3:2.2f}ms")
    t0 = time.time()
    # copy results over to CPU space
    intensity, fcoords, dcoords = cp.asnumpy(intensity), cp.asnumpy(fcoords), cp.asnumpy(dcoords)
    t1 = time.time()
    logger.info(f"Peak find memcopy: {(t1-t0)*1e3:2.2f}ms")
    
    t0 = time.time()
    if len(fcoords) > 0:
        driftrate_peaks = drift_trials[dcoords]
        logger.debug(f"{metadata['fch1']}, {metadata['df']}, {fcoords}")
        frequency_peaks = metadata['fch1'] + metadata['df'] * fcoords


        results = {
            'driftrate': driftrate_peaks,
            'f_start': frequency_peaks,
            'snr': intensity,
            'driftrate_idx': dcoords,
            'channel_idx': fcoords
        }

        # Append numerical metadata keys
        for key, val in metadata.items():
            if isinstance(val, (int, float)):
                results[key] = val

        return pd.DataFrame(results)
        t1 = time.time()
        logger.info(f"Peak find to dataframe: {(t1-t0)*1e3:2.2f}ms")
    else:
        return None
    
    
def merge_hits(hitlist):
    """ Group hits corresponding to different boxcar widths and return hit with max SNR 
    
    Args:
        hitlist (pd.DataFrame): List of hits
    
    Returns:
        hitlist (pd.DataFrame): Abridged list of hits after merging
    """
    t0 = time.time()
    p = hitlist.sort_values('snr', ascending=False)
    hits = []
    while len(p) > 1:
        # Grab top hit 
        p0 = p.iloc[0]

        # Find channels and driftrates within tolerances
        q = f"""(abs(driftrate_idx - {p0['driftrate_idx']}) <= boxcar_size + 1  |
                abs(driftrate_idx - {p0['driftrate_idx']}) <= {p0['boxcar_size']} + 1)
                & 
                (abs(channel_idx - {p0['channel_idx']}) <= {p0['boxcar_size']} + 1| 
                abs(channel_idx - {p0['channel_idx']}) <= boxcar_size + 1)"""
        q = q.replace('\n', '') # Query must be one line
        pq = p.query(q)
        tophit = pq.sort_values("snr", ascending=False).iloc[0]

        # Drop all matched rows
        p = p.drop(pq.index)
        hits.append(tophit)
    t1 = time.time()
    logger.info(f"Hit merging time: {(t1-t0)*1e3:2.2f}ms")
    
    return pd.DataFrame(hits)


def run_pipeline(data, metadata, max_dd, min_dd=None, threshold=50, min_fdistance=None, 
                 min_ddistance=None, n_boxcar=6, merge_boxcar_trials=True, apply_normalization=False):
    """ Run dedoppler + hitsearch pipeline 
    
    Args:
        data (np.array): Numpy array with shape (N_timestep, N_channel)
        metadata (dict): Metadata dictionary, should contain 'df' and 'dt'
                         (frequency and time resolution), as astropy quantities
        max_dd (float): Maximum doppler drift in Hz/s to search out to.
        min_dd (float): Minimum doppler drift to search.
        n_boxcar (int): Number of boxcar trials to do, width 2^N e.g. trials=(1,2,4,8,16)
        merge_boxcar_trials (bool): Merge hits of boxcar trials to remove 'duplicates'. Default True.
        apply_normalization (bool): Normalize input data. Default True. Required True for S/N calcs.
        threshold (float): Threshold value (absolute) above which a peak can be considered
        min_fdistance (int): Minimum distance in pixels to nearest peak along frequency axis
        min_ddistance (int): Minimum distance in pixels to nearest peak along doppler axis
    
    Returns:
        (dedopp, metadata, peaks): Array of data post dedoppler (at final boxcar width), plus
                                   metadata (dict) and then table of hits (pd.Dataframe).
    """
    
    t0 = time.time()
    logger.debug(data.shape)
    N_timesteps = data.shape[0]
    _threshold = threshold * np.sqrt(N_timesteps)
    
    # Apply preprocessing normalization
    if apply_normalization:
        data = normalize(data, return_space='gpu')
    
    
    peaks = create_empty_hits_table()
    
    boxcar_trials = map(int, 2**np.arange(0, n_boxcar))
    for boxcar_size in boxcar_trials:
        logger.info(f"--- Boxcar size: {boxcar_size} ---")
        dedopp, metadata = dedoppler(data, metadata, boxcar_size=boxcar_size,  boxcar_mode='sum',
                                     max_dd=max_dd, min_dd=min_dd, return_space='gpu')
        
        # Adjust SNR threshold to take into account boxcar size and dedoppler sum
        # Noise increases by sqrt(N_timesteps * boxcar_size)
        _threshold = threshold * np.sqrt(N_timesteps * boxcar_size)
        _peaks = hitsearch(dedopp, metadata, threshold=_threshold, min_fdistance=min_fdistance, min_ddistance=min_ddistance)
        
        if _peaks is not None:
            _peaks['snr'] /= np.sqrt(N_timesteps * boxcar_size)
            peaks = pd.concat((peaks, _peaks), ignore_index=True)
            
    if merge_boxcar_trials:
        peaks = merge_hits(peaks)
    t1 = time.time()
    
    logger.info(f"Pipeline runtime: {(t1-t0):2.2f}s")
    return dedopp, metadata, peaks
            
    
def find_et_serial(filename, filename_out='hits.csv', gulp_size=2**19, *args, **kwargs):
    """ Find ET, serial version
    
    Wrapper for reading from a file and running run_pipeline() on all subbands within the file.
    
    Args:
        filename (str): Name of input HDF5 file.
        filename_out (str): Name of output CSV file.
        gulp_size (int): Number of channels to process at once (e.g. N_chan in a coarse channel)
    
    Returns:
        hits (pd.DataFrame): Pandas dataframe of all hits.
    
    Notes:
        Passes keyword arguments on to run_pipeline(). Same as find_et but doesn't use dask parallelization.
    """
    t0 = time.time()
    #peaks = create_empty_hits_table()    
    ds = from_h5(filename)
    out = []
    for d_arr in ds.iterate_through_data({'frequency': gulp_size}):
        print(d_arr)
        d = d_arr.data
        f = d_arr.frequency
        t = d_arr.time
        md = {'fch1': f.val_start * f.units, 'df': f.val_step * f.units, 'dt': t.val_step * t.units}
        dedopp, metadata, hits = run_pipeline(d, md, *args, **kwargs)
        out.append(hits)
        logger.info(f"{len(hits)} hits found")
    
    dframe = pd.concat(out)
    dframe.to_csv(filename_out)
    t1 = time.time()
    print(f"## TOTAL TIME: {(t1-t0):2.2f}s ##\n\n")
    return dframe