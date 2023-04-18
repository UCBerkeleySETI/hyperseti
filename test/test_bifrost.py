from hyperseti.hyperfrost.bifrost import from_bf, to_bf
from hyperseti.hyperfrost.data_source import DataArrayBlock
from hyperseti.test_data import voyager_fil, voyager_h5

import bifrost as bf
import bifrost.pipeline as bfp

import numpy as np
from datetime import datetime
from pprint import pprint

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
        if self.n_iter % 100 == 0:
            print("[%s] %s" % (now, ispan.data))
        self.n_iter += 1


def test_pipeline():

    filenames   = [voyager_fil, ]

    b_read      = DataArrayBlock(filenames, 32768, gulp_nframe=1, axis='frequency', overlap=0)
    b_print     = PrintStuffBlock(b_read)

    # Run pipeline
    pipeline = bfp.get_default_pipeline()
    print(pipeline.dot_graph())
    pipeline.run()

if __name__ == "__main__":
    test_pipeline()
    # test_from_bf()
