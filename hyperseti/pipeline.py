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
from .utils import attach_gpu_device, timeme
from .blanking import blank_edges, blank_extrema
from .hit_browser import HitBrowser
from hyperseti.version import HYPERSETI_VERSION

from hyperseti.data_array import DataArray

#logging
from .log import get_logger
import logbook
logger = get_logger('hyperseti.pipeline')

proglog = get_logger('find_et')
proglog.level = logbook.INFO

# Max threads setup 
os.environ['NUMEXPR_MAX_THREADS'] = '8'

class GulpPipeline(object):
    """ Pipeline class for a single channel or 'gulp' 

    This class is called 
    
    Provides the following methods:
        pipeline.preprocess() - Preprocess data
        pipeline.dedoppler()  - Apply dedoppler transform
        pipeline.hitsearch()  - Search dedoppler space for hits 
        pipeline.run()        - Run all pipeline stages 
                                This loops over boxcar trials / blanking iterations
    
    Inputs/outputs can be accessed via:
        pipeline.config       - Input dict to configure pipeline at init
        pipeline.data_array   - Input data array
        pipeline.dedopp       - Output dedoppler array
        pipeline.dedopp_sk    - Output ddsk array (if selected)
        pipeline.peaks        - Local maxima (hits) output of hitsearch()

    Example usage:
        ```
        pipeline = GulpPipeline(d_arr, config)
        pipeline.run()
        ```
    """

    def __init__(self, data_array: DataArray, config: dict, gpu_id: int=None):
        """ Pipeline class to run on a gulp of data (e.g. a coarse channel)

        Args:
            data_array (DataArray): Data Array object with dims (time, beam_id, frequency)
            config (dict): Dictionary of config values. Dictionary values are passed as 
                           keyword arguments to called functions. 
            gpu_id (int): Choose GPU to run pipeline on by integer ID (default None)
        """
        self.data_array = data_array
        self.config = deepcopy(config)
        self._called_count = 0

        if not isinstance(self.data_array.data, cp.ndarray):
            logger.warning(f"GulpPipeline init: Data not in cupy.ndarray, attempting to copy data to GPU")
            self.data_array.data = cp.asarray(self.data_array.data)
        
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
        n_boxcar = config['pipeline'].get('n_boxcar', 1) 
        n_boxcar = 1 if n_boxcar is None else n_boxcar
        if config['dedoppler']['apply_smearing_corr'] and n_boxcar > 1:
            logger.warning("GulpPipeline: combining dedoppler/apply_smearing_corr and pipeline/n_boxcar > 1 may produce strange results. Or it may not. Not sure yet.")

    @timeme
    def preprocess(self):
        """ Apply preprocessing steps 
        
        Preprocessing steps:
            1) Blank edge channels of gulp (optional)
            2) Flag non-gaussian data before computing stats (optional)
            3) Normalize data (subtract mean and convert into units of SNR)
            4) Blank any stupidly bright channels (optional)
        """
        
        # Apply main normalization
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

        # Blank edges 
        if self.config['preprocess'].get('blank_edges', 0):
            logger.info(f"GulpPipeline.preprocess: Applying edge blanking")
            self.data_array = blank_edges(self.data_array, **self.config['preprocess']['blank_edges'])

        # Extrema blanking is done *after* normalization
        if self.config['preprocess'].get('blank_extrema'):
            logger.info(f"GulpPipeline.preprocess: Blanking extremely bright signals")
            self.data_array = blank_extrema(self.data_array, **self.config['preprocess']['blank_extrema'])

        # Proglog info
        pp_dict = self.data_array.attrs['preprocess']
        proglog.info(f"\tPreprocess mean:       {pp_dict['mean']}")
        proglog.info(f"\tPreprocess STD:        {pp_dict['std']}")
        proglog.info(f"\tPreprocess flagged:    {pp_dict['flagged_fraction']:.2f}%")

    @timeme        
    def dedoppler(self):
        """ Apply dedoppler transform to gulp """
        # Check if kernel is computing DD + SK
        kernel = self.config['dedoppler'].get('kernel', None)
        if kernel == 'ddsk':
            logger.info("GulpPipeline.dedoppler: Running DDSK dedoppler kernel")
            self.dedopp, self.dedopp_sk = dedoppler(self.data_array, **self.config['dedoppler'])
        else:
            logger.info("GulpPipeline.dedoppler: Running standard dedoppler kernel")
            self.dedopp = dedoppler(self.data_array,  **self.config['dedoppler'])
            self.dedopp_sk = None
    
    @timeme
    def hitsearch(self):
        """ Run search for hits above threshold in dedoppler space.
        
        Notes:
            self.dedoppler() must be called first, otherwise there's nothing to search.
        """
        # Adjust SNR threshold to take into account boxcar size and dedoppler sum
        # Noise increases by sqrt(N_timesteps * boxcar_size)
        # sqrt(N_timesteps) is taken into account within dedoppler kernel
        conf = deepcopy(self.config)        # This deepcopy avoids overwriting original threshold value
        boxcar_size = conf['dedoppler'].get('boxcar_size', 1)
        _threshold0 = conf['hitsearch']['threshold']
        conf['hitsearch']['threshold'] = _threshold0 * np.sqrt(boxcar_size)
        _peaks = hitsearch(self.dedopp, **conf['hitsearch'])
        
        if _peaks is not None:
            _peaks['snr'] /= np.sqrt(boxcar_size)
            proglog.info(f"\t Hits in gulp: {len(_peaks)}")
            self.peaks = pd.concat((self.peaks, _peaks), ignore_index=True)   

    @timeme   
    def run(self) -> pd.DataFrame:
        """ Main pipeline runner 
        
        Returns:
            self.peaks (pd.DataFrame): Table of all hits
        """

        self._called_count += 1
        t0 = time.time()

        self.preprocess()

        n_boxcar = self.config['pipeline'].get('n_boxcar', 1)
        n_blank  = self.config['pipeline'].get('n_blank', 1)
        n_boxcar = 1 if n_boxcar is None else n_boxcar     # None value breaks loop
        n_blank  = 1 if n_blank is None else n_blank       # None value breaks loop
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

        if self._called_count == 1:    
            logger.info(f"GulpPipeline.run: Elapsed time: {(t1-t0):2.2f}s; {len(self.peaks)} hits found")
        else:
            logger.info(f"GulpPipeline.run #{self._called_count}: Elapsed time: {(t1-t0):2.2f}s; {len(self.peaks)} hits found")
        return self.peaks

@timeme
def find_et(filename: str, 
            pipeline_config: dict, 
            filename_out: str='hits.csv', 
            gulp_size: int=2**20, 
            sort_hits: bool=True,
            log_config: bool=False,
            log_output: bool=False,
            gpu_id: int=0, 
            *args, **kwargs) -> pd.DataFrame:
    """ Find ET, serial version
    
    Wrapper for reading from a file and running run_pipeline() on all subbands within the file.
    
    Args:
        filename (str): Name of input HDF5 file.
        pipeline_config (dict): See findET.py command-line parameters.
        filename_out (str): Name of output CSV file.
        log_output (bool): If set, will log pipeline output to TXT file.
        log_config (bool): If set, will log pipeline configuration to YAML file.
        sort_hits (bool): Sort hits by SNR after hitsearch is complete.
        gulp_size (int): Number of channels to process in one 'gulp' ('gulp' can be == 'coarse channel')
        gpu_id (int): GPU device ID to use.
   
    Returns:
        hits (pd.DataFrame): Pandas dataframe of all hits.
    
    Notes:
        Passes keyword arguments on to GulpPipeline.run(). 
    """
    if log_output:
        from logbook import FileHandler, NestedSetup
        from .log import log_to_screen
        logfile_out = os.path.splitext(filename_out)[0] + '.log'
        log_to_file = FileHandler(logfile_out, bubble=True)
        logger_setup = NestedSetup([log_to_screen, log_to_file])
        logger_setup.push_application()
    
    if log_config:
        import yaml
        config_out = os.path.splitext(filename_out)[0] + '.yaml'
        with open(config_out, 'w') as json_out:
            yaml.dump(pipeline_config, json_out)

    msg = f"find_et: hyperseti version {HYPERSETI_VERSION}"
    proglog.info(msg)
    #print(msg)
    logger.info(pipeline_config)

    if isinstance(filename, DataArray):
        ds = filename           # User has actually supplied a DataArray
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
    n_gulps = ds.data.shape[-1] // gulp_size
    for d_arr in ds.iterate_through_data({'frequency': gulp_size}, space='gpu'):
        counter += 1
        proglog.info(f"Progress {counter}/{n_gulps}")
        pipeline = GulpPipeline(d_arr, pipeline_config, gpu_id=gpu_id)
        hits = pipeline.run()
        out.append(hits)
    
    dframe = pd.concat(out)

    if sort_hits:
        dframe = dframe.sort_values('snr', ascending=False).reset_index(drop=True)

    if filename_out is not None:
        dframe.to_csv(filename_out, index=False)

    if log_config:
        print(f"find_et: Pipeline runtime config logged to {config_out}")

    if log_output:
        print(f"find_et: Output logged to {logfile_out}")

    return HitBrowser(ds, dframe)
    