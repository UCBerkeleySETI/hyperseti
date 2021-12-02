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
from .data_array import from_fil, from_h5
from .utils import on_gpu, datwrapper

#logging
import logbook
from .log import logger_group, Logger
logger = Logger('hyperseti.hyperseti')
logger_group.add_logger(logger)

# Max threads setup 
os.environ['NUMEXPR_MAX_THREADS'] = '8'


@datwrapper(dims=(None))
@on_gpu
def run_pipeline(data, metadata, max_dd, min_dd=None, threshold=50, min_fdistance=None, 
                 min_ddistance=None, n_boxcar=6, merge_boxcar_trials=True, apply_normalization=True,
                 gpu_id=0, log_level=logbook.INFO):
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
        gpu_id (int): GPU device ID to use
        log_level (int): logbook level to use
    
    Returns:
        (dedopp, metadata, peaks): Array of data post dedoppler (at final boxcar width), plus
                                   metadata (dict) and then table of hits (pd.Dataframe).
    """
    
    t0 = time.time()
    logger_group.level = log_level
    attach_gpu_device(gpu_id)
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
        dedopp, md = dedoppler(data, metadata, boxcar_size=boxcar_size,  boxcar_mode='sum',
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
    
    logger.info(f"Pipeline runtime: {(t1-t0):2.2f}s")
    return peaks
            

def find_et(filename, filename_out='hits.csv', gulp_size=2**19, gpu_id=0, log_level=logbook, *args, **kwargs):
    """ Find ET, serial version
    
    Wrapper for reading from a file and running run_pipeline() on all subbands within the file.
    
    Args:
        filename (str): Name of input HDF5 file.
        filename_out (str): Name of output CSV file.
        gulp_size (int): Number of channels to process at once (e.g. N_chan in a coarse channel)
        gpu_id (int): GPU device ID to use
        log_level (int): logbook level to use
   
    Returns:
        hits (pd.DataFrame): Pandas dataframe of all hits.
    
    Notes:
        Passes keyword arguments on to run_pipeline(). Same as find_et but doesn't use dask parallelization.
    """
    t0 = time.time()
    logger_group.level = log_level
    #peaks = create_empty_hits_table()    
    ds = from_h5(filename)
    out = []
    for d_arr in ds.iterate_through_data({'frequency': gulp_size}):
        #print(d_arr)
        hits = run_pipeline(d_arr, gpu_id=gpu_id, log_level=log_level, *args, **kwargs)
        out.append(hits)
        logger.info(f"{len(hits)} hits found")
    
    dframe = pd.concat(out)
    dframe.to_csv(filename_out)
    t1 = time.time()
    print(f"## TOTAL TIME: {(t1-t0):2.2f}s ##\n\n")
    return dframe
