import pandas as pd
import numpy as np
import cupy as cp
import os
from datetime import datetime
from pprint import pprint
from copy import deepcopy

import bifrost as bf
import bifrost.pipeline as bfp

from hyperseti.hits import create_empty_hits_table
from hyperseti.pipeline import GulpPipeline
from hyperseti.data_array import DataArray, from_metadata
from hyperseti.io.hit_db import HitDatabase
from hyperseti.kernels import DedopplerMan, PeakFinderMan, SmearCorrMan, BlankHitsMan

from hyperseti.hyperfrost.bifrost import from_bf

from hyperseti.log import get_logger
logger = get_logger('hyperfrost.pipeline')

def strip_frame_dimension(data_array: DataArray) -> DataArray:
        """ Remove outer 'frame' dimension from bifrost-iterated array 
        
        Basically applies a squeeze(axis=0) and removes some metadata

        Return:
            data_array (DataArray): A data array with outer 'frame' dimension stripped
        """
        data, metadata = data_array.split_metadata()
        data = data.squeeze(axis=0)

        metadata.pop('frame_step')
        metadata.pop('frame_start')
        metadata['dims'] = list(metadata['dims'])[1:]
        return from_metadata(data, metadata)


class HyperfrostPipelineBlock(bfp.SinkBlock):
    def __init__(self, iring, pipeline_config: dict, db: str, log_level: str=None, N_workers: int=1, worker_id: int=0, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)

        self.config = pipeline_config
        self.db = db
        self._reset()
        self._worker_id = worker_id
        self._N_workers = N_workers
        
        if log_level is not None:
            logger.level = log_level

        with HitDatabase(self.db, mode='w') as db:
             logger.info(f"Creating h5 file {db}")

        self.kernel_managers = {
            'dedoppler': DedopplerMan(),
            'blank_hits': BlankHitsMan(),
            'peak_find': PeakFinderMan(),
            'smear_corr': SmearCorrMan()
        }
        
    def _reset(self):
        self.n_iter = 0
        self.out = []
        self.iseq_header = None
        
    def on_sequence(self, iseq):
        self._reset()
        logger.info("[%s]" % datetime.now())
        logger.info(iseq.name)
        logger.info(iseq.header)
        self.iseq_header = iseq.header    

    def on_data(self, ispan):
            
        hdr = deepcopy(self.iseq_header)
        for key_to_ignore in ('time_tag', 'gulp_nframe'):
            hdr.pop(key_to_ignore)
        
        if self._N_workers > 1:
            data = bf.ndarray(np.expand_dims(np.copy(ispan.data[self._worker_id]), axis=0))
        else:
            data = ispan.data

        d_arr = from_bf(data, self.iseq_header)
        d_arr = strip_frame_dimension(d_arr)
        #print("HERE", self.n_iter, self._worker_id)
        pipeline = GulpPipeline(d_arr, self.config, kernel_managers=self.kernel_managers)
        hits = pipeline.run()
        
        # Add some runtime info
        if len(hits) > 0:
            hits['gulp_idx']     = self.n_iter
            hits['gulp_size']    = d_arr.data.shape[2]
            hits['hits_in_gulp'] = len(hits)

            if 'preprocess' in d_arr.attrs.keys():
                for beam_idx in range(d_arr.shape[1]):
                    hits[f'b{beam_idx}_gulp_mean'] = cp.asnumpy(d_arr.attrs['preprocess']['mean'][beam_idx])
                    hits[f'b{beam_idx}_gulp_std']  = cp.asnumpy(d_arr.attrs['preprocess']['std'][beam_idx])
                    hits[f'b{beam_idx}_gulp_flag_frac'] = cp.asnumpy(d_arr.attrs['preprocess'].get('flagged_fraction', 0))
                

                if d_arr.attrs['preprocess'].get('n_poly', 0) > 1:
                    n_poly = d_arr.attrs['preprocess']['n_poly']
                    hits['n_poly'] = n_poly
                    for pp in range(int(n_poly + 1)):
                        hits[f'b{beam_idx}_gulp_poly_c{pp}'] = cp.asnumpy(d_arr.attrs['preprocess']['poly_coeffs'][beam_idx, pp])

            self.out.append(hits)
        self.n_iter += 1

    def on_sequence_end(self, iseq):
        if len(self.out) == 0:
            dframe = create_empty_hits_table()
        else:
            dframe = pd.concat(self.out)
        print(dframe)

        with HitDatabase(self.db, mode='r+') as db:
                obs_id = self.iseq_header['name']
                logger.info(f"Saving output to {db}/{obs_id}")
                db.add_obs(obs_id, 
                            hit_table=dframe, 
                            input_filename=self.iseq_header.get('filepath', None), 
                            config=self.config)

def hyperfrost_pipeline(iring, N_workers: int, pipeline_config: dict, db: str, log_level: str=None, *args, **kwargs):
    db_base, db_ext = os.path.splitext(db)
    b_gulp = bf.blocks.copy(iring, gulp_nframe=N_workers, buffer_nframe=N_workers*2, buffer_factor=N_workers*2)
    for w_id in range(N_workers):
        os.path.splitext(db)[0]
        w_db = f"{db_base}_{w_id}.{db_ext}"
        b_hyper     = HyperfrostPipelineBlock(b_gulp, pipeline_config, db, log_level=log_level, N_workers=N_workers, worker_id=w_id, core=w_id+1, *args, **kwargs)
    