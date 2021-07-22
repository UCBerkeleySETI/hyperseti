from hyperseti.data_array import from_fil, from_h5
from .file_defs import voyager_fil, voyager_h5

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

if __name__ == "__main__":
    test_load_voyager()
    test_data_array_basic()