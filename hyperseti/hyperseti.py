import cupy as cp
import numpy as np
import time
import pandas as pd
import os
import h5py

from astropy import units as u
import setigen as stg
from blimpy.io import sigproc
from copy import deepcopy

from .dedoppler import dedoppler
from .normalize import normalize
from .hits import hitsearch, merge_hits, create_empty_hits_table, blank_hits
from .io import from_fil, from_h5
from .utils import attach_gpu_device, on_gpu, datwrapper
from .kurtosis import sk_flag
from hyperseti.version import HYPERSETI_VERSION

#logging
from .log import get_logger
logger = get_logger('hyperseti.hyperseti')

# Max threads setup 
os.environ['NUMEXPR_MAX_THREADS'] = '8'


def run_pipeline(data_array, config, gpu_id=0, called_count=None):
    """ Run dedoppler + hitsearch pipeline 
    
    Args:
        data (np.array): Numpy array with shape (N_timestep, N_channel)
        metadata (dict): Metadata dictionary, should contain 'frequency_step' and 'time_step'
                         (frequency and time resolution), as astropy quantities
        config (dict): See findET.py command-line parameters.
        gpu_id (int): GPU device ID to use.
        called_count (int): If called in a loop, use this to keep track of how many times the
                            pipeline is called. 
    
    Returns:
        (dedopp, metadata, peaks): Array of data post dedoppler (at final boxcar width), plus
                                   metadata (dict) and then table of hits (pd.Dataframe).
    """
    config = deepcopy(config)
    t0 = time.time()
    if gpu_id is not None:
        attach_gpu_device(gpu_id)

    data = data_array.data
    metadata = data_array.metadata
    N_timesteps = data.shape[0]

    logger.debug(f"run_pipeline: data.shape={data.shape}, metadata={metadata}")

    # Check if we have a slice 
    
    # Calculate order for argrelmax (minimum distance from peak)
    min_fdistance = config['hitsearch'].get('min_fdistance', None)
    max_dd = config['dedoppler']['max_dd']
    if min_fdistance is None:
        deltaf = metadata['frequency_step'].to('Hz').value
        deltat = metadata['time_step'].to('s').value
        n_int  = data.shape[0]

        min_fdistance = int(np.abs(deltat * n_int * max_dd / deltaf))
        config['hitsearch']['min_fdistance'] = min_fdistance
        logger.debug(f"run_pipeline: min_fdistance calculated to be {min_fdistance} bins")

    # Apply preprocessing normalization
    if config['preprocess'].get('normalize', False):
        poly_fit = config['preprocess'].get('poly_fit', 0)
        if config['preprocess'].get('sk_flag', False):
            sk_flag_opts = config.get('sk_flag', {})
            mask = sk_flag(data_array, **sk_flag_opts)
        else:
            mask = None
        data_array = normalize(data_array, mask=mask, poly_fit=poly_fit)
    
    peaks = create_empty_hits_table()
    
    n_boxcar = config['pipeline'].get('n_boxcar', 1)
    n_blank = config['pipeline'].get('n_blank', 1)
    boxcar_trials = map(int, 2**np.arange(0, n_boxcar))
    
    _threshold0 = deepcopy(config['hitsearch']['threshold'])
    n_hits_last_iter = 0
    for blank_count in range(n_blank):
        for boxcar_size in boxcar_trials:
            logger.debug(f"run_pipeline: --- Boxcar size: {boxcar_size} ---")
            config['dedoppler']['boxcar_size'] = boxcar_size
            
            # Check if kernel is computing DD + SK
            kernel = config['dedoppler'].get('kernel', None)
            if kernel == 'ddsk':
                dedopp, dedopp_sk = dedoppler(data_array, **config['dedoppler'])
                config['hitsearch']['sk_data'] = dedopp_sk   # Pass to hitsearch
            else:
                dedopp = dedoppler(data_array,  **config['dedoppler'])
                dedopp_sk = None
            
            # Adjust SNR threshold to take into account boxcar size and dedoppler sum
            # Noise increases by sqrt(N_timesteps * boxcar_size)
            config['hitsearch']['threshold'] = _threshold0 * np.sqrt(N_timesteps * boxcar_size)
            _peaks = hitsearch(dedopp, **config['hitsearch'])
            logger.debug(f"{peaks}")
            
            if _peaks is not None:
                _peaks['snr'] /= np.sqrt(N_timesteps * boxcar_size)
                peaks = pd.concat((peaks, _peaks), ignore_index=True)   
            else:
                peaks = peaks

        if config['pipeline']['merge_boxcar_trials']:
            peaks = merge_hits(peaks)
        n_hits_iter = len(peaks) - n_hits_last_iter

        if n_blank > 1:
            if n_hits_iter > n_hits_last_iter:
                logger.info(f"run_pipeline: blanking hits, (iteration {blank_count + 1} / {n_blank})")
                data, metadata = blank_hits(data, metadata, peaks)
                data = data.data ## Blank hits will return a DataArray
                n_hits_last_iter = n_hits_iter
            else:
                logger.info(f"run_pipeline: No new hits found, breaking! (iteration {blank_count + 1} / {n_blank})")
                break

    t1 = time.time()

    if called_count is None:    
        logger.info(f"run_pipeline: Elapsed time: {(t1-t0):2.2f}s; {len(peaks)} hits found")
    else:
        logger.info(f"run_pipeline #{called_count}: Elapsed time: {(t1-t0):2.2f}s; {len(peaks)} hits found")

    return peaks
            

def find_et(filename, pipeline_config, filename_out='hits.csv', gulp_size=2**20, gpu_id=0, *args, **kwargs):
    """ Find ET, serial version
    
    Wrapper for reading from a file and running run_pipeline() on all subbands within the file.
    
    Args:
        filename (str): Name of input HDF5 file.
        pipeline_config (dict): See findET.py command-line parameters.
        filename_out (str): Name of output CSV file.
        gulp_size (int): Number of channels to process at once (e.g. N_chan in a coarse channel)
        gpu_id (int): GPU device ID to use.
   
    Returns:
        hits (pd.DataFrame): Pandas dataframe of all hits.
    
    Notes:
        Passes keyword arguments on to run_pipeline(). Same as find_et but doesn't use dask parallelization.
    """
    msg = f"find_et: hyperseti version {HYPERSETI_VERSION}"
    logger.info(msg)
    print(msg)
    logger.info(pipeline_config)
    t0 = time.time()

    if h5py.is_hdf5(filename):
        ds = from_h5(filename)
    elif sigproc.is_filterbank(filename):
        ds = from_fil(filename)
    elif isinstance(stg.Frame, filename):
        ds = filename 
    else:
        raise RuntimeError("Only HDF5 and filterbank files currently supported")

    if gulp_size > ds.data.shape[2]:
        logger.warning(f'find_et: gulp_size ({gulp_size}) > Num fine frequency channels ({ds.data.shape[2]}).  Setting gulp_size = {ds.data.shape[2]}')
        gulp_size = ds.data.shape[2]
    out = []

    attach_gpu_device(gpu_id)
    counter = 0
    for d_arr in ds.iterate_through_data({'frequency': gulp_size}, space='gpu'):
        counter += 1
        hits = run_pipeline(d_arr, pipeline_config, gpu_id=None, called_count=counter)
        out.append(hits)
    
    dframe = pd.concat(out)
    dframe.to_csv(filename_out)
    t1 = time.time()
    msg = f"find_et: TOTAL ELAPSED TIME: {(t1-t0):2.2f}s"
    logger.info(msg)
    print(msg)
    return dframe
