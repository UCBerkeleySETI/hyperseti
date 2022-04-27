import cupy as cp
import numpy as np
import time
import pandas as pd
import os

from astropy import units as u
import setigen as stg

from cupyx.scipy.ndimage import uniform_filter1d

from .dedoppler import dedoppler
from .normalize import normalize
from .filter import apply_boxcar
from .hits import hitsearch, merge_hits, create_empty_hits_table
from .peak import prominent_peaks
from .io import from_fil, from_h5
from .utils import attach_gpu_device, on_gpu, datwrapper
from .kurtosis import sk_flag

#logging
from .log import get_logger
logger = get_logger('hyperseti.hyperseti')

# Max threads setup 
os.environ['NUMEXPR_MAX_THREADS'] = '8'


@datwrapper(dims=(None))
@on_gpu
def run_pipeline(data, metadata, called_count=None, max_dd=4.0, min_dd=0.001, threshold=30.0, min_fdistance=None, 
                 min_ddistance=None, n_boxcar=6, merge_boxcar_trials=True, apply_normalization=True,
                 kernel='dedoppler', gpu_id=0):
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
        kernel (str): which GPU kernel to employ for searching.
        gpu_id (int): GPU device ID to use.
    
    Returns:
        (dedopp, metadata, peaks): Array of data post dedoppler (at final boxcar width), plus
                                   metadata (dict) and then table of hits (pd.Dataframe).
    """
    
    t0 = time.time()
    if gpu_id is not None:
        attach_gpu_device(gpu_id)
    logger.debug(data.shape)
    N_timesteps = data.shape[0]
    _threshold = threshold * np.sqrt(N_timesteps)
    
    # Apply preprocessing normalization
    if apply_normalization:
        mask = sk_flag(data, metadata, return_space='gpu')
        data = normalize(data, mask=mask, return_space='gpu')
    
    peaks = create_empty_hits_table()
    
    boxcar_trials = map(int, 2**np.arange(0, n_boxcar))
    for boxcar_size in boxcar_trials:
        logger.debug(f"run_pipeline: --- Boxcar size: {boxcar_size} ---")
        dedopp, md = dedoppler(data, metadata, boxcar_size=boxcar_size,  boxcar_mode='sum', kernel=kernel,
                                     max_dd=max_dd, min_dd=min_dd, return_space='gpu')
        
        # Adjust SNR threshold to take into account boxcar size and dedoppler sum
        # Noise increases by sqrt(N_timesteps * boxcar_size)
        _threshold = threshold * np.sqrt(N_timesteps * boxcar_size)
        _peaks = hitsearch(dedopp, threshold=_threshold, min_fdistance=min_fdistance, min_ddistance=min_ddistance)
        
        if _peaks is not None:
            _peaks['snr'] /= np.sqrt(N_timesteps * boxcar_size)
            peaks = pd.concat((peaks, _peaks), ignore_index=True)
            
    if merge_boxcar_trials:
        peaks = merge_hits(peaks)
    t1 = time.time()

    if called_count is None:    
        logger.info(f"run_pipeline: Elapsed time: {(t1-t0):2.2f}s; {len(peaks)} hits found")
    else:
        logger.info(f"run_pipeline #{called_count}: Elapsed time: {(t1-t0):2.2f}s; {len(peaks)} hits found")

    return peaks
            

def find_et(filename, filename_out='hits.csv', gulp_size=2**19, max_dd=4.0, min_dd=0.001, 
            min_fdistance=None, min_ddistance=None, threshold=30.0,
            n_boxcar=6, kernel='dedoppler', gpu_id=0, *args, **kwargs):
    """ Find ET, serial version
    
    Wrapper for reading from a file and running run_pipeline() on all subbands within the file.
    
    Args:
        filename (str): Name of input HDF5 file.
        filename_out (str): Name of output CSV file.
        gulp_size (int): Number of channels to process at once (e.g. N_chan in a coarse channel)
        max_dd (float): Maximum doppler drift in Hz/s to search out to.
        min_dd (float): Minimum doppler drift to search.
        threshold (float): Minimum SNR value to use in a search.
        n_boxcar (int): Number of boxcar trials to do, width 2^N e.g. trials=(1,2,4,8,16).
        kernel (str): which GPU kernel to employ for searching.
        gpu_id (int): GPU device ID to use.
   
    Returns:
        hits (pd.DataFrame): Pandas dataframe of all hits.
    
    Notes:
        Passes keyword arguments on to run_pipeline(). Same as find_et but doesn't use dask parallelization.
    """
    t0 = time.time()
    logger.info("find_et: At entry, filename_out={}, gulp_size={}, max_dd={}, min_dd={}, threshold={}, n_boxcar={}, kernel={}, gpu_id={}"
                 .format(filename_out, gulp_size, max_dd, min_dd, threshold, n_boxcar, kernel, gpu_id))
    ds = from_h5(filename)

    if min_fdistance is None:
        deltaf = ds.frequency.to('Hz').val_step
        deltat = ds.time.to('s').val_step
        min_fdistance = int(np.abs(deltat * ds.time.n_step * max_dd / deltaf))
        logger.info(f"find_et: min_fdistance calculated to be {min_fdistance} bins")

    if gulp_size > ds.data.shape[2]:
        logger.warning(f'find_et: gulp_size ({gulp_size}) > Num fine frequency channels ({ds.data.shape[2]}).  Setting gulp_size = {ds.data.shape[2]}')
        gulp_size = ds.data.shape[2]
    out = []
    attach_gpu_device(gpu_id)
    counter = 0
    for d_arr in ds.iterate_through_data({'frequency': gulp_size}):
        counter += 1
        hits = run_pipeline(d_arr, called_count=counter, max_dd=max_dd, min_dd=min_dd, 
                                threshold=threshold, n_boxcar=n_boxcar, 
                                min_fdistance=min_fdistance, min_ddistance=min_ddistance,
                                kernel=kernel, gpu_id=None, *args, **kwargs)
        out.append(hits)
    
    dframe = pd.concat(out)
    dframe.to_csv(filename_out)
    t1 = time.time()
    print(f"find_et: TOTAL TIME: {(t1-t0):2.2f}s ##\n\n")
    return dframe