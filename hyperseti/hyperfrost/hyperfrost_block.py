import pandas as pd
import numpy as np
from datetime import datetime
from pprint import pprint
import cupy as cp

import bifrost.pipeline as bfp

from hyperseti.hits import create_empty_hits_table
from hyperseti.pipeline import GulpPipeline
from hyperseti.data_array import DataArray, from_metadata

from hyperseti.hyperfrost.bifrost import from_bf

from hyperseti.log import get_logger
logger = get_logger('hyperfrost.pipeline')

def strip_frame_dimension(data_array: DataArray) -> DataArray:
        data, metadata = data_array.split_metadata()
        data = data.squeeze(axis=0)

        metadata.pop('frame_step')
        metadata.pop('frame_start')
        metadata['dims'] = list(metadata['dims'])[1:]
        return from_metadata(data, metadata)


class HyperfrostPipelineBlock(bfp.SinkBlock):
    def __init__(self, iring, pipeline_config: dict, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)

        self.config = pipeline_config
        self._reset()
        
    def _reset(self):
        self.n_iter = 0
        self.out = []
        self.iseq_header = None
        
    def on_sequence(self, iseq):
        self._reset()
        #print("[%s]" % datetime.now())
        #print(iseq.name)
        #pprint(iseq.header)
        self.iseq_header = iseq.header

    def on_data(self, ispan):

        d_arr = from_bf(ispan.data, self.iseq_header)
        d_arr = strip_frame_dimension(d_arr)
        print(d_arr, 'HERE')

        pipeline = GulpPipeline(d_arr, self.config)
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

        self.n_iter += 1
        
        self.out.append(hits)

    def on_sequence_end(self, iseq):
        if len(self.out) == 0:
            dframe = create_empty_hits_table()
        else:
            dframe = pd.concat(self.out)
        print(dframe)