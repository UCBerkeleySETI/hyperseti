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

from .dedoppler import dedoppler, calc_ndrift
from .normalize import normalize
from .hits import hitsearch, merge_hits, create_empty_hits_table, blank_hits
from .io import from_fil, from_h5, from_setigen
from .kurtosis import sk_flag
from .utils import attach_gpu_device
from .blanking import blank_edges, blank_extrema
from hyperseti.version import HYPERSETI_VERSION

#logging
from .log import get_logger
logger = get_logger('hyperseti.pipeline')

# Max threads setup 
os.environ['NUMEXPR_MAX_THREADS'] = '8'

class GulpPipeline(object):
    def __init__(self, data_array, config, gpu_id=None):
        self.data_array = data_array
        self.config = deepcopy(config)
        
        if gpu_id is not None:
            attach_gpu_device(gpu_id)
        self.gpu_id = gpu_id

        self.data = data_array.data
        self.metadata = data_array.metadata
        self.N_timesteps = self.data.shape[0]

        logger.debug(f"GulpPipeline init: data.shape={self.data.shape}, metadata={self.metadata}")

        self.peaks = create_empty_hits_table()

        # Calculate order for argrelmax (minimum distance from peak)
        min_fdistance = self.config['hitsearch'].get('min_fdistance', None)
        max_dd = self.config['dedoppler']['max_dd']
        if min_fdistance is None:
            min_fdistance = calc_ndrift(data_array, max_dd)
            self.config['hitsearch']['min_fdistance'] = min_fdistance
            logger.info(f"GulpPipeline: min_fdistance calculated to be {min_fdistance} bins")

        # TODO: Do rigorous testing of interplay between apply_smearing_corr and boxcar_width
        if config['dedoppler']['apply_smearing_corr'] and config['pipeline']['n_boxcar'] > 1:
            logger.warning("GulpPipeline: combining dedoppler/apply_smearing_corr and pipeline/n_boxcar > 1 may produce strange results. Or it may not. Not sure yet.")

    def preprocess(self):
        # Apply preprocessing normalization and blanking
        if self.config['preprocess'].get('blank_edges', 0):
            logger.info(f"GulpPipeline.preprocess: Applying edge blanking")
            self.data_array = blank_edges(self.data_array, **self.config['preprocess']['blank_edges'])
        
        if self.config['preprocess'].get('normalize', False):
            poly_fit = self.config['preprocess'].get('poly_fit', 0)
            if self.config['preprocess'].get('sk_flag', False):
                logger.info(f"GulpPipeline.preprocess: Applying sk_flag before normalization")
                sk_flag_opts = self.config.get('sk_flag', {})
                mask = sk_flag(self.data_array, **sk_flag_opts)
            else:
                mask = None

            logger.info(f"GulpPipeline.preprocess: Normalizing data")
            self.data_array = normalize(self.data_array, mask=mask, poly_fit=poly_fit)
            self.mask = mask

        # Extrema blanking is done *after* normalization
        if self.config['preprocess'].get('blank_extrema'):
            logger.info(f"GulpPipeline.preprocess: Blanking extremely bright signals")
            self.data_array = blank_extrema(self.data_array, **self.config['preprocess']['blank_extrema'])
            
    def dedoppler(self):
            # Check if kernel is computing DD + SK
            kernel = self.config['dedoppler'].get('kernel', None)
            if kernel == 'ddsk':
                logger.info("GulpPipeline.dedoppler: Running DDSK dedoppler kernel")
                self.dedopp, self.dedopp_sk = dedoppler(self.data_array, **self.config['dedoppler'])
            else:
                logger.info("GulpPipeline.dedoppler: Running standard dedoppler kernel")
                self.dedopp = dedoppler(self.data_array,  **self.config['dedoppler'])
                self.dedopp_sk = None
    
    def hitsearch(self):
        # Adjust SNR threshold to take into account boxcar size and dedoppler sum
        # Noise increases by sqrt(N_timesteps * boxcar_size)
        conf = deepcopy(self.config)        # This deepcopy avoids overwriting original threshold value
        boxcar_size = conf['dedoppler']['boxcar_size']
        _threshold0 = conf['hitsearch']['threshold']
        conf['hitsearch']['threshold'] = _threshold0 * np.sqrt(self.N_timesteps * boxcar_size)
        _peaks = hitsearch(self.dedopp, **conf['hitsearch'])

        #logger.debug(f"{self.peaks}")
        
        if _peaks is not None:
            _peaks['snr'] /= np.sqrt(self.N_timesteps * boxcar_size)
            self.peaks = pd.concat((self.peaks, _peaks), ignore_index=True)   
        
    def run(self, called_count=None):
        t0 = time.time()

        self.preprocess()
    
        n_boxcar = self.config['pipeline'].get('n_boxcar', 1)
        n_blank  = self.config['pipeline'].get('n_blank', 1)
        boxcar_trials = list(map(int, 2**np.arange(0, n_boxcar)))
    
        n_hits_last_iter = 0
        for blank_count in range(n_blank):
            for boxcar_idx, boxcar_size in enumerate(boxcar_trials):
                logger.debug(f"GulpPipeline.run: boxcar_size {boxcar_size}, ({boxcar_idx + 1} / {len(boxcar_trials)})")
                self.config['dedoppler']['boxcar_size'] = boxcar_size
                self.dedoppler()
                self.hitsearch()

            n_hits_iter = len(self.peaks) - n_hits_last_iter
            logger.debug(f"GulpPipeline.run: New hits: {n_hits_iter}")

            if self.config['hitsearch'].get('merge_boxcar_trials', True):
                logger.info(f"GulpPipeline.run: merging hits")
                self.peaks = merge_hits(self.peaks)

           
            if n_blank > 1:
                if n_hits_iter > n_hits_last_iter:
                    logger.info(f"GulpPipeline.run: blanking hits, (iteration {blank_count + 1} / {n_blank})")
                    self.data_array = blank_hits(self.data_array, self.peaks)
                    n_hits_last_iter = n_hits_iter
                else:
                    logger.info(f"GulpPipeline.run: No new hits found, breaking! (iteration {blank_count + 1} / {n_blank})")
                    break

        t1 = time.time()

        if called_count is None:    
            logger.info(f"GulpPipeline.run: Elapsed time: {(t1-t0):2.2f}s; {len(self.peaks)} hits found")
        else:
            logger.info(f"GulpPipeline.run #{called_count}: Elapsed time: {(t1-t0):2.2f}s; {len(self.peaks)} hits found")

        return self.peaks

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
        pipeline = GulpPipeline(d_arr, pipeline_config, gpu_id=gpu_id)
        hits = pipeline.run()
        out.append(hits)
    
    dframe = pd.concat(out)
    dframe.to_csv(filename_out)
    t1 = time.time()
    msg = f"find_et: TOTAL ELAPSED TIME: {(t1-t0):2.2f}s"
    logger.info(msg)
    print(msg)
    return dframe
    

