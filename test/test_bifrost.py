from hyperseti.hyperfrost.bifrost import from_bf, to_bf
from hyperseti.hyperfrost.data_source import DataArrayBlock
from hyperseti.hyperfrost.hyperfrost_block import HyperfrostPipelineBlock
from hyperseti.test_data import voyager_fil, voyager_h5

import bifrost as bf
import bifrost.pipeline as bfp

import numpy as np
from datetime import datetime
from pprint import pprint

from hyperseti.log import get_logger
logger = get_logger('hyperfrost.data_source', 'info')
logger = get_logger('hyperfrost.pipeline', 'info')

def test_from_bf():
    d = bf.ndarray(np.zeros((10, 2, 1000)), space='cuda', dtype='f32')

    ohdr = {
        '_tensor': {
            'dtype':  'f32',
            'shape':  d.shape,
            'labels': ['time', 'pol', 'freq'],
            'scales': [[0, 0.1],  None, (1420.0, 0.01)],
            'units':  ['s', None, 'MHz'] }
    }

    d_out = from_bf(d, ohdr)
    print(d_out)

    d_roundtrip, ohdr_roundtrip = to_bf(d_out)
    print(d_roundtrip.shape, d_roundtrip.dtype)
    print(ohdr_roundtrip)

    t_rt = ohdr_roundtrip['_tensor']
    assert t_rt['labels'] == ['time', 'pol', 'freq']
    assert t_rt['dtype']  == 'f32'
    assert t_rt['units'] == ['s', '', 'MHz']  # Note: None is converted into ''



class PrintStuffBlock(bfp.SinkBlock):
    def __init__(self, iring, *args, **kwargs):
        super(PrintStuffBlock, self).__init__(iring, *args, **kwargs)
        self.n_iter = 0

    def on_sequence(self, iseq):
        print("[%s]" % datetime.now())
        print(iseq.name)
        pprint(iseq.header)
        self.n_iter = 0

    def on_data(self, ispan):
        now = datetime.now()
        if self.n_iter % 10 == 0:
            print("[%s] %s" % (now, ispan.data.shape))
        self.n_iter += 1


def test_pipeline():

    filenames   = [voyager_h5, voyager_fil]

    b_read      = DataArrayBlock(filenames, 32768, axis='frequency', overlap=0)
    b_print     = PrintStuffBlock(b_read)

    # Run pipeline
    pipeline = bfp.get_default_pipeline()
    print(pipeline.dot_graph())
    pipeline.run()

def test_pipeline_overlap():

    filenames   = [voyager_h5, voyager_fil]

    b_read      = DataArrayBlock(filenames, 32768, axis='frequency', overlap=1000)
    b_print     = PrintStuffBlock(b_read)

    # Run pipeline
    pipeline = bfp.get_default_pipeline()
    print(pipeline.dot_graph())
    pipeline.run()

def test_pipeline_full():
    filenames   = [voyager_h5, voyager_fil]

    pipeline_config = {
        'preprocess': {
            'sk_flag': True,
            'normalize': True,
            'poly_fit': 3,
            'blank_edges': {'n_chan': 1024},
            'blank_extrema': {'threshold': 10000},
        },
        'dedoppler': {
            'kernel': 'dedoppler',
            'max_dd': 4.0,
            'min_dd': None,
            'apply_smearing_corr': True,
            'plan': 'stepped',
            },
        'hitsearch': {
            'threshold': 20,
            'min_fdistance': None,
            'merge_boxcar_trials': True
        },
        'pipeline': {
            'n_boxcar': 1,
            'n_blank': 4,
        }
    }

    b_read      = DataArrayBlock(filenames, 2**20, axis='frequency', overlap=0)
    b_hyper     = HyperfrostPipelineBlock(b_read, pipeline_config, db='test.hitdb')

    # Run pipeline
    pipeline = bfp.get_default_pipeline()
    print(pipeline.dot_graph())
    pipeline.run()

def test_cmd_tool():
    

if __name__ == "__main__":
    test_pipeline_full()
    #test_pipeline()
    #test_pipeline_overlap()
    # test_from_bf()
