from hyperseti.hyperfrost.bifrost import from_bf, to_bf
import bifrost as bf
import numpy as np

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

if __name__ == "__main__":
    d = test_from_bf()
