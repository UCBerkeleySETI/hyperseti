import cupy as cp
import numpy as np
import time
import pandas as pd
import os
import h5py

from astropy import units as u
import setigen as stg
from blimpy.io import sigproc

from .dedoppler import dedoppler
from .normalize import normalize
from .hits import hitsearch, merge_hits, create_empty_hits_table
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
def run_pipeline(data, metadata, max_dd=4.0, min_dd=None, threshold=30.0, min_fdistance=None, 
                 n_boxcar=1, merge_boxcar_trials=True, apply_normalization=True,
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
    attach_gpu_device(gpu_id)
    logger.debug(data.shape)
    N_timesteps = data.shape[0]
    _threshold = threshold * np.sqrt(N_timesteps)
    
    if min_fdistance is None:
        print(metadata)
        deltaf = metadata['frequency_step'].to('Hz').value
        deltat = metadata['time_step'].to('s').value
        n_int  = data.shape[0]

        min_fdistance = int(np.abs(deltat * n_int * max_dd / deltaf))
        logger.info(f"<run_pipeline>: min_fdistance calculated to be {min_fdistance} bins")

    # Apply preprocessing normalization
    if apply_normalization:
        mask = sk_flag(data, metadata, return_space='gpu')
        data = normalize(data, mask=mask, return_space='gpu')
    
    peaks = create_empty_hits_table()
    
    boxcar_trials = map(int, 2**np.arange(0, n_boxcar))
    for boxcar_size in boxcar_trials:
        logger.info(f"--- Boxcar size: {boxcar_size} ---")
        dedopp, md = dedoppler(data, metadata, boxcar_size=boxcar_size,  boxcar_mode='sum', kernel=kernel,
                                     max_dd=max_dd, min_dd=min_dd, return_space='gpu')
        
        # Adjust SNR threshold to take into account boxcar size and dedoppler sum
        # Noise increases by sqrt(N_timesteps * boxcar_size)
        _threshold = threshold * np.sqrt(N_timesteps * boxcar_size)
        _peaks = hitsearch(dedopp, threshold=_threshold, min_fdistance=min_fdistance)
        
        if _peaks is not None:
            _peaks['snr'] /= np.sqrt(N_timesteps * boxcar_size)
            peaks = pd.concat((peaks, _peaks), ignore_index=True)
            
    if merge_boxcar_trials:
        peaks = merge_hits(peaks)
    t1 = time.time()
    
    logger.info(f"Pipeline runtime: {(t1-t0):2.2f}s")
    return peaks
            

def find_et(filename, filename_out='hits.csv', gulp_size=2**19, max_dd=4.0, min_dd=0.001, 
            min_fdistance=None,  threshold=20.0,
            n_boxcar=1, kernel='dedoppler', gpu_id=0, *args, **kwargs):
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
    logger.debug("find_et: At entry, filename_out={}, gulp_size={}, max_dd={}, min_dd={}, threshold={}, n_boxcar={}, kernel={}, gpu_id={}"
                 .format(filename_out, gulp_size, max_dd, min_dd, threshold, n_boxcar, kernel, gpu_id))
    if h5py.is_hdf5(filename):
        ds = from_h5(filename)
    elif sigproc.is_filterbank(filename):
        ds = from_fil(filename)
    else:
        raise RuntimeError("Only HDF5 and filterbank files currently supported")

    if gulp_size > ds.data.shape[2]:
        logger.warning(f'find_et: gulp_size ({gulp_size}) > Num fine frequency channels ({ds.data.shape[2]}).  Setting gulp_size = {ds.data.shape[2]}')
        gulp_size = ds.data.shape[2]
    out = []

    for d_arr in ds.iterate_through_data({'frequency': gulp_size}):
        #print(d_arr)
        hits = run_pipeline(d_arr, max_dd=max_dd, min_dd=min_dd, 
                                threshold=threshold, n_boxcar=n_boxcar, 
                                min_fdistance=min_fdistance, kernel=kernel, 
                                gpu_id=gpu_id, *args, **kwargs)
        out.append(hits)
        logger.info(f"{len(hits)} hits found")
    
    dframe = pd.concat(out)
    dframe.to_csv(filename_out)
    t1 = time.time()
    print(f"## TOTAL TIME: {(t1-t0):2.2f}s ##\n\n")
    return dframe