from hyperseti.dimension_scale import DimensionScale, TimeScale
import numpy as np
from astropy import units as u
import pytest

def test_dimension_scale():
    print("\n-------TESTING DimensionScale-------")
    ds = DimensionScale('frequency', 1000, 0.1, n_step=100, units=u.GHz)
    print(ds)
    ds = DimensionScale('frequency', 1000, 0.1, n_step=100, units='GHz')
    print(ds)

    print("SHAPE", ds.shape)
    print("LEN", len(ds))
    print("NDIM", ds.ndim)

    # Check conversion to array
    print(np.asarray(ds))
    # print(np.duckarray(ds)) NOT YET IMPLEMENTED
    ds_arr = ds.generate(0, 10)
    print(ds_arr)


    # Check fancy indexing
    print("--INDEXING---")
    print(ds[10:20])
    print(ds[10:20:2])
    print(np.asarray(ds[10:20:2]))
    print(ds_arr[5:9])
    print(ds[6])

    # Check reverse indexing
    print(ds.index(1008))
    print(ds.index(1008, 1010))
    print(ds.index(np.arange(1000, 1010)))

    # Check addition / sub /mult /div
    ds1 = DimensionScale('frequency', 1000, 0.1, n_step=100, units='MHz')
    ds2 = DimensionScale('frequency', 1, 0.001, n_step=100, units='GHz')
    ds3 = DimensionScale('frequency', 2, 0.001, n_step=100, units='GHz')
    
    ds_new = ds1 + ds2
    try:
        ds1 +  DimensionScale('raise_error', 1, 1, n_step=1001, units='GHz')
    except ValueError:
        print("Caught non-matching lengths error")

    try:
        ds1 + DimensionScale('raise_error', 1, 1, n_step=100, units='Jy')
    except ValueError:
        print("Caught incompatible unit addition error")

    print (ds_new)
    assert ds_new.val_start == 2000.0
    assert ds_new.units == u.MHz

    ds_new = ds3 - ds2
    print(ds_new)
    assert ds_new.val_start == 1.0
    assert ds_new.units == u.GHz

    ds_new = ds2 * ds3
    print(ds_new)
    ds_new = ds2 / ds3
    print(ds_new)

    # Check unit conversion
    ds_new = ds1.to('Hz')
    print(ds_new)

def test_scalar_add():
    print("\n-------TESTING DimensisonScale Arithmetic-------")
    ds1 = DimensionScale('frequency', 1000, 0.1, n_step=100, units='MHz')
    print(ds1)
    print(ds1 + 9)
    print(ds1 + (9*u.MHz))
    print(ds1 + (9000*u.kHz))

    print(ds1 - 9)
    print(ds1 - (9*u.MHz))
    print(ds1 - (9000*u.kHz))
    
    print(ds1 * 9)
    print(ds1 * (9*u.MHz))
    print(ds1 * (9000*u.kHz))

    print(ds1 / 9)
    print(ds1 / (9*u.MHz))
    print(ds1 / (9000*u.kHz))
    
def test_time_scale():
    print("\n-------TESTING TimeScale-------")
    # Values from Voyager test data in blimpy/turboseti repo
    tstart = 57650.78209490741  # MJD
    tsamp  = 18.253611008       # Seconds
    
    ts = TimeScale('time', tstart, tsamp, 16, time_format='mjd', time_delta_format='sec')
    print(ts)
    print(np.asarray(ts))
    print(ts.generate())
    print(ts[0:8:2])
    print(ts[0:8:2].generate())
    print(ts[0])
    
    # Subtracting times will give a DimensionScale
    # Although this isn't something that makes much sense to do in reality
    ts1 = TimeScale('time', tstart, tsamp, 16, time_format='mjd', time_delta_format='sec')
    ts2 = TimeScale('time', tstart + 1, tsamp, 16, time_format='mjd', time_delta_format='sec')
    print(ts2 - ts1)


def test_raises():
    print("\n-------TESTING Error Raising-------")
    ds = DimensionScale('frequency', 1000, 0.1, n_step=100, units='MHz')
    ts = TimeScale('time', 60000, 1.0, 16, time_format='mjd', time_delta_format='sec')

    with pytest.raises(IndexError):
        ds[100000]
        ts[100000]
    
    with pytest.raises(TypeError): 
        # TypeError: no implementation found for 'numpy.mean' on types that
        # implement __array_function__: [<class 'hyperseti.dimension_scale.DimensionScale'>]
        np.mean(ds)
    
    with pytest.raises(ValueError):
        ds.index(-1)
        ds.index(np.arange(-1, 1, 0.1))
        ds.index(1001, -9)
  
def test_index():
    # Test index() method works with quantities
    ds1 = DimensionScale('frequency', 1000, 0.1, n_step=100, units='MHz')
    q = 1001 * u.MHz

    print(ds1.index(q))
    q = np.array([1000, 1001, 1002.11]) * 1e6 * u.Hz
    print(ds1.index(q))
    assert np.allclose(ds1.index(q), [0, 10, 21])


if __name__ == "__main__":
    test_dimension_scale()
    test_time_scale()
    test_scalar_add()
    test_raises()
    test_index()
