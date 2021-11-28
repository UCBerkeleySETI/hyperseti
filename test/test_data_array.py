from hyperseti.data_array import from_fil, from_h5
from .file_defs import voyager_fil, voyager_h5
import numpy as np
import cupy as cp
import pytest

def test_load_voyager():
    d = from_fil(voyager_fil)

    print(d)
    print(d.attrs)
    print(d.dims)
    print(d.scales)
    print(d.data.shape)

    d = from_h5(voyager_h5)

    print(d)
    print(d.attrs)
    print(d.dims)
    print(d.scales)
    print(d.data.shape)

def test_data_array_basic():
    d = from_h5(voyager_h5)
    
    # Test slice data
    sel = d.isel({'frequency': slice(8192, 8192+4096)})
    assert sel.shape == (16, 1, 4096)
    
    # Test iterate through chunks
    counter = 0
    for ds_sub in d.iterate_through_data({'frequency': 1048576 // 16, 'time': 16 // 2}):
        counter += 1
        assert ds_sub.shape == (8, 1, 65536)
        
    assert counter == 16 * 2
    
    assert d.shape == (16, 1, 1048576)
    assert str(d.dtype) == 'float32'
    
    # Test __repr__ and html output
    print(d)
    html = d._repr_html_()

def test_asarray():
    d = from_h5(voyager_h5)
    d_np = np.asarray(d)
    assert isinstance(d_np, np.ndarray)
    
    d = from_fil(voyager_fil)
    d_np = np.asarray(d)
    assert isinstance(d_np, np.ndarray)
    
    d.data = cp.array(d.data)
    d_np = np.asarray(d)
    assert isinstance(d_np, np.ndarray)    
    
    # Test applying random numpy ufunc
    d = from_h5(voyager_h5)
    d_log = np.log(d)
    assert isinstance(d_log, np.ndarray)
    
def test_apply_transform():
    d = from_h5(voyager_h5)
    d_copy = np.copy(d.data[:])
    
    d.apply_transform('sqrt')
    assert np.allclose(d, np.sqrt(d_copy))
    d.apply_transform('power', 2)
    assert np.allclose(d, d_copy)
    
    # test function
    def sqrt(x):
        return np.sqrt(x)
    d.apply_transform(sqrt)
    assert np.allclose(d, sqrt(d_copy))
    
    # Test cupy
    d.data = cp.array(d.data)
    d.apply_transform('log2')
    
    def unlog2(x):
        return cp.power(2, x)
    d.apply_transform(unlog2)
    
    with pytest.raises(RuntimeError):
        d.apply_transform('chicken') # This should fail
    
if __name__ == "__main__":
    test_load_voyager()
    test_data_array_basic()
    test_asarray()
    test_apply_transform()
