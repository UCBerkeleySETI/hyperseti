import cupy as cp
import numpy as np
import time
import pandas as pd
import logging
import os

from astropy import units as u
import setigen as stg

from cupyimg.skimage.feature.peak import _prominent_peaks as prominent_peaks
from cupyx.scipy.ndimage import uniform_filter1d

import hdf5plugin
import h5py
from copy import deepcopy

from multiprocessing.pool import ThreadPool
import dask
import dask.bag as db
from dask.diagnostics import ProgressBar

# Max threads setup 
os.environ['NUMEXPR_MAX_THREADS'] = '8'
MAX_THREADS = 4
dask.config.set(pool=ThreadPool(MAX_THREADS))

# Logger setup
logger_name = 'hyperseti'
logger = logging.getLogger(logger_name)
logger.setLevel(logging.CRITICAL)
#logger.setLevel(logging.INFO)

dedoppler_kernel = cp.RawKernel(r'''
extern "C" __global__
    __global__ void dedopplerKernel
        (const float *data, float *dedopp, int *shift, int F, int T)
        /* Each thread computes a different dedoppler sum for a given channel
        
         F: N_frequency channels
         T: N_time steps
        
         *data: Data array, (T x F) shape
         *dedopp: Dedoppler summed data, (D x F) shape
         *shift: Array of doppler corrections of length D.
                 shift is total number of channels to shift at time T
        */
        {
        
        // Setup thread index
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int d   = blockIdx.y;   // Dedoppler trial ID
        const int D   = gridDim.y;   // Number of dedoppler trials

        // Index for output array
        const int dd_idx = d * F + tid;
        int idx = 0;
        
        for (int t = 0; t < T; t++) {
                            // timestep    // dedoppler trial offset
            idx  = tid + (F * t)      + (shift[d] * t / T);
            if (idx < F * T && idx > 0) {
                dedopp[dd_idx] += data[idx];
              }
            }
        }
''', 'dedopplerKernel')


def normalize(data, return_space='cpu'):
    """ Apply normalization on GPU
    
    Applies normalisation (data - median) / stdev
    
    Args: 
        data (np/cp.array): Data to preprocess
        return_space ('cpu' or 'gpu'): Returns array in CPU or GPU space
        
    Returns: d_gpu (cp.array): Preprocessed data
    """
    
    # Copy over to GPU if required
    d_gpu = cp.asarray(data.astype('float32'))
    
    # Normalise
    t0 = time.time()
    d_median = cp.median(d_gpu)
    d_std  = cp.std(d_gpu)
    d_gpu = (d_gpu - d_median) / d_std
    t1 = time.time()
    logger.info(f"Normalisation time: {(t1-t0)*1e3:2.2f}ms")
    
    if return_space == 'cpu':
        return cp.asnumpy(d_gpu)
    else:
        return d_gpu

def apply_boxcar(data, boxcar_size, axis=1, mode='mean', return_space='cpu'):
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
    data = cp.asarray(data.astype('float32'))
    data = uniform_filter1d(data, size=boxcar_size, axis=axis)
    if mode == 'gaussian':
        data *= np.sqrt(boxcar_size)
    elif mode == 'sum':
        data *= boxcar_size
    t1 = time.time()
    logger.info(f"Filter time: {(t1-t0)*1e3:2.2f}ms")
    
    if return_space == 'cpu':
        return cp.asnumpy(data)
    else:
        return data
    
def dedoppler(data, metadata, max_dd, min_dd=None, boxcar_size=1,
              boxcar_mode='sum', return_space='cpu'):
    """ Apply brute-force dedoppler kernel to data
    
    Args:
        data (np.array): Numpy array with shape (N_timestep, N_channel)
        metadata (dict): Metadata dictionary, should contain 'df' and 'dt'
                         (frequency and time resolution)
        max_dd (float): Maximum doppler drift in Hz/s to search out to
        min_dd (float): Minimum doppler drift to search
        
        apply_normalization (bool): Apply preprocessing to normalise data. Default False
        boxcar_mode (str): Boxcar mode to apply. mean/sum/gaussian.
        return_space ('cpu' or 'gpu'): Returns array in CPU or GPU space
    
    Returns:
        dd_vals, dedopp_gpu (np.array, np/cp.array): 
    """
    if min_dd is None:
        min_dd = np.abs(max_dd) * -1
    
    # Compute minimum possible drift (delta_dd)
    N_time, N_chan = data.shape
    obs_len  = N_time * metadata['dt'].to('s').value
    delta_dd = metadata['df'].to('Hz').value / obs_len  # e.g. 2.79 Hz / 300 s = 0.0093 Hz/s
    
    # Compute dedoppler shift schedules
    N_dopp_upper   = int(max_dd / delta_dd)
    N_dopp_lower   = int(min_dd / delta_dd)
    
    if N_dopp_upper > N_dopp_lower:
        dd_shifts      = np.arange(N_dopp_lower, N_dopp_upper + 1, dtype='int32')
    else:
        dd_shifts      = np.arange(N_dopp_upper, N_dopp_lower + 1, dtype='int32')[::-1]
        
    dd_shifts_gpu  = cp.asarray(dd_shifts)
    N_dopp = len(dd_shifts)
    
    # Copy data over to GPU
    d_gpu = cp.asarray(data.astype('float32'))
    
    # Apply boxcar filter
    if boxcar_size > 1:
        d_gpu = apply_boxcar(d_gpu, boxcar_size, mode='sum', return_space='gpu')
    
    # Allocate GPU memory for dedoppler data
    dedopp_gpu = cp.zeros((N_dopp, N_chan), dtype=cp.float32)
    
    # Setup grid and block dimensions
    F_block = np.min((N_chan, 1024))
    F_grid  = N_chan // F_block
    
    # Launch kernel
    t0 = time.time()
    #print(dd_shifts)
    logger.debug("Kernel shape (grid, block)", (F_grid, N_dopp), (F_block,))
    dedoppler_kernel((F_grid, N_dopp), (F_block,), 
                     (d_gpu, dedopp_gpu, dd_shifts_gpu, N_chan, N_time)) # grid, block and arguments
        
    t1 = time.time()
    logger.info(f"Kernel time: {(t1-t0)*1e3:2.2f}ms")
    
    # Compute drift rate values in Hz/s corresponding to dedopp axis=0
    dd_vals = dd_shifts * delta_dd
    
    metadata['drift_trials'] = dd_vals
    metadata['boxcar_size'] = boxcar_size
    metadata['dd'] = delta_dd * u.Hz / u.s
    
    # Copy back to CPU if requested
    if return_space == 'cpu':
        dedopp_cpu = cp.asnumpy(dedopp_gpu)
        return dedopp_cpu, metadata
    else:
        return dedopp_gpu, metadata

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
    
    # Unfortunately we need a CPU copy of the dedoppler for peak finding
    # This means lots of copying back and forth, so potential bottleneck
    if isinstance(dedopp, np.ndarray):
        dedopp_cpu = dedopp
    else:
        dedopp_cpu = cp.asnumpy(dedopp)
        
    # Copy over to GPU if required
    dedopp_gpu = cp.asarray(dedopp.astype('float32'))
    
    t0 = time.time()
    intensity, fcoords, dcoords = prominent_peaks(dedopp_gpu, min_xdistance=min_fdistance, min_ydistance=min_ddistance, threshold=threshold)
    # copy results over to CPU space
    intensity, fcoords, dcoords = cp.asnumpy(intensity), cp.asnumpy(fcoords), cp.asnumpy(dcoords)
    t1 = time.time()
    logger.info(f"Peak find time: {(t1-t0)*1e3:2.2f}ms")
    
    if len(fcoords) > 0:
        driftrate_peaks = drift_trials[dcoords]
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
        q = f"""(abs(driftrate_idx - {p0['driftrate_idx']}) <= boxcar_size  |
                abs(driftrate_idx - {p0['driftrate_idx']}) <= {p0['boxcar_size']})
                & 
                (abs(channel_idx - {p0['channel_idx']}) <= {p0['boxcar_size']} | 
                abs(channel_idx - {p0['channel_idx']}) <= boxcar_size)"""
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
    """ Run pipeline """
    
    t0 = time.time()
    N_timesteps = data.shape[0]
    _threshold = threshold * np.sqrt(N_timesteps)
    
    # Apply preprocessing normalization
    if apply_normalization:
        data = normalize(data, return_space='gpu')
    
    # Create empty dataframe
    peaks = pd.DataFrame({'driftrate': pd.Series([], dtype='float64'),
                          'f_start': pd.Series([], dtype='float64'),
                          'snr': pd.Series([], dtype='float64'),
                          'driftrate_idx': pd.Series([], dtype='int'),
                          'channel_idx': pd.Series([], dtype='int'),
                          'boxcar_size': pd.Series([], dtype='int'),
                         })
    
    boxcar_trials = map(int, 2**np.arange(0, n_boxcar))
    for boxcar_size in boxcar_trials:
        logger.info(f"--- Boxcar size: {boxcar_size} ---")
        dedopp, metadata = dedoppler(data, metadata, boxcar_size=boxcar_size,  boxcar_mode='sum',
                                     max_dd=max_dd, min_dd=min_dd)
        
        # Adjust SNR threshold to take into account boxcar size and dedoppler sum
        # Noise increases by sqrt(N_timesteps * boxcar_size)
        _threshold = threshold * np.sqrt(N_timesteps * boxcar_size)
        _peaks = hitsearch(dedopp, metadata, threshold=_threshold, min_fdistance=min_fdistance, min_ddistance=min_ddistance)
        _peaks['snr'] /= np.sqrt(N_timesteps * boxcar_size)
        
        # Join to list
        peaks = pd.concat((peaks, _peaks), ignore_index=True)
            
    if merge_boxcar_trials:
        peaks = merge_hits(peaks)
    t1 = time.time()
    
    logger.info(f"Pipeline runtime: {(t1-t0):2.2f}s")
    return dedopp, metadata, peaks
            
class H5Reader(object):
    """ Basic HDF5 reader """
    def __init__(self, fn, gulp_size=2**19):
        self.fn = fn
        
        with h5py.File(fn, mode='r') as h:
            self.metadata = {
                'fch1': h['data'].attrs['fch1'] * u.MHz,
                'dt': h['data'].attrs['tsamp'] * u.s,
                'df': (h['data'].attrs['foff'] * u.MHz).to('Hz')
            }
            self.dshape = h['data'].shape
            
        self.gulp_size = 2**19
        self.n_sub = self.dshape[2] // gulp_size
    
    def read_data_plan(self):
        for ii in range(self.n_sub):
            i0, i1 = ii*self.gulp_size, (ii+1)*self.gulp_size
            md = deepcopy(self.metadata)
            md['fch1'] += md['df'] * i0
            md['i0'] = i0
            md['i1'] = i1
            md['sidx'] = ii
            yield md

    def read_data(self, md):
        t0 = time.time()
        ii = md['sidx']
        with h5py.File(self.fn, mode='r') as h:
            d = h['data'][:, 0, md['i0']:md['i1']]
        t1 = time.time()
        logger.info(f"## Subband {ii+1}/{self.n_sub} read: {(t1-t0)*1e3:2.2f}ms ##")
        return d

def search_subband_dask(md, h5):
    d_gulp = h5.read_data(md)
    dd, dedopp, peaks = run_pipeline(d_gulp, md, max_dd=1.0, min_dd=None, 
                                     threshold=50, min_fdistance=100, min_ddistance=None, 
                                     n_boxcar=5, merge_boxcar_trials=True)
    if not peaks.empty:
        peaks['channel_idx'] += md['i0']
    return [peaks,]

def find_et(filename, filename_out='hits.csv', n_parallel=1, gulp_size=2**19):
    
    h5 = H5Reader(filename, gulp_size=gulp_size)   
    
    dask.config.set(pool=ThreadPool(n_parallel))
    t0 = time.time()
    b = db.from_sequence(h5.read_data_plan())
    with ProgressBar():
        logger.setLevel(logging.CRITICAL)
        out = b.map(search_subband_dask, h5).compute()

    dframe = pd.concat([o[0] for o in out])
    dframe.to_csv(filename_out)
    t1 = time.time()
    print(f"## N_PARALLEL {n_parallel} TOTAL TIME: {(t1-t0):2.2f}s ##\n\n")
    return dframe
    
if __name__ == "__main__":
    fn = '/datax/collate_mb/PKS_0262_2018-02-21T17:00/blc01/guppi_58171_08035_757812_G26.37-1.21_0001.0000.hires.hdf'
    h5 = H5Reader(fn, gulp_size=2**20)

    for MAX_THREADS in (1,2,3,4):
        dask.config.set(pool=ThreadPool(MAX_THREADS))

        t0 = time.time()
        b = db.from_sequence(h5.read_data_plan())
        with ProgressBar():
            logger.setLevel(logging.CRITICAL)
            out = b.map(search_subband_dask, h5).compute()

        dframe = pd.concat([o[0] for o in out])
        dframe.to_csv('hits.csv')
        t1 = time.time()
        print(f"## MAXTHREADS {MAX_THREADS} TOTAL TIME: {(t1-t0):2.2f}s ##\n\n")
