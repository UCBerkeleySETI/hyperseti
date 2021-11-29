import numpy as np
import cupy as cp
import pandas as pd
from hyperseti.dedoppler import dedoppler
from hyperseti.data_array import DataArray, from_h5, from_fil
from hyperseti.dimension_scale import DimensionScale, TimeScale
from hyperseti.utils import datwrapper, on_gpu
from .file_defs import voyager_fil, voyager_h5
import pytest

import logbook
import hyperseti
hyperseti.utils.logger.level = logbook.DEBUG

def test_datwrapper_basic():
    """ Basic test of data/metadata vs DataArray handling """
    ## Simple wrapper that shouldn't do anything
    @datwrapper(dims=None)
    def do_something(data, metadata):
        return data

    # check numpy array input + metadata
    data = np.array([1,2,3,4])
    metadata = {'frequency_step': 1}
    data_out = do_something(data, metadata)
    assert isinstance(data_out, np.ndarray)

    # Check cupy array + metadata
    data = cp.array([1,2,3,4])
    metadata = {'frequency_step': 1}
    data_out = do_something(data, metadata)
    assert isinstance(data_out, cp.core.core.ndarray)

    ## Check DataArray works without metadata
    data_arr = from_fil(voyager_fil)
    do_something(data_arr)

def test_datwrapper_warnings():
    data = np.array([1,2,3,4])
    metadata = {'frequency_step': 1}
    
    ## Wrapper that will create a RuntimeWarning as it can't create DataArray from bare numpy 
    @datwrapper(dims=('pork', 'noodles'))
    def do_something_warn_np(data, metadata):
        return data
    with pytest.warns(RuntimeWarning):
        do_something_warn_np(data, metadata)

    ## Wrapper that will create a RuntimeWarning as it can't create DataArray from pandas dataframe
    @datwrapper(dims=('pork', 'noodles'))
    def do_something_warn_pd(data, metadata):
        return pd.DataFrame() 
    with pytest.warns(RuntimeWarning):
        do_something_warn_pd(data, metadata)

    ## Wrapper that will create a RuntimeWarning as it can't create DataArray from pandas dataframe
    @datwrapper(dims=('pork', 'noodles'))
    def do_something_warn_int(data, metadata):
        return 1 
    with pytest.warns(RuntimeWarning):
        do_something_warn_int(data, metadata)

def test_datwrapper_create_data_array():
    @datwrapper(dims=('pork', 'noodles'))
    def do_create_data_array(data, metadata):
        new_metadata = {'pork_start': 1,
                    'pork_step': 0.1,
                    'noodles_start': 2,
                    'noodles_step': 0.2}
        metadata.update(new_metadata)
        new_data = np.zeros(shape=(10, 10))
        return new_data, metadata
    
    data = np.array([1,2,3,4])
    metadata = {'test': 0}
    
    dout, mdout = do_create_data_array(data, metadata)
    assert isinstance(dout, DataArray)
    assert isinstance(mdout, dict)
    assert dout.dims == ('pork', 'noodles')
    assert len(dout.scales.keys()) == 2
    assert 'test' in mdout.keys()
    assert 'test' in dout.attrs.keys()
    assert len(mdout) == 6    # test, pork_start, pork_step, noodles_start, noodles_Step, output_dims
    assert len(dout.attrs) == 1   # Check step/stop are parsed as dimensions
    
    dout2, mdout2 = do_create_data_array(dout)
    assert isinstance(dout2, DataArray)
    assert isinstance(mdout2, dict)
    assert dout2.dims == ('pork', 'noodles')
    assert len(dout2.scales.keys()) == 2
    assert len(mdout2) == 6    # test, pork_start, pork_step, noodles_start, noodles_Step, output_dims ## input_dims
    assert len(dout2.attrs) == 1   # Check step/stop are parsed as dimensions
    
    dout2.attrs['test2'] = 'hi'
    dout3, mdout3 = do_create_data_array(dout2)
    assert isinstance(dout3, DataArray)
    assert isinstance(mdout3, dict)
    assert 'test2' in mdout3.keys()
    assert 'test2' in dout3.attrs.keys()
    assert len(mdout3) == 7    # test, pork_start, pork_step, noodles_start, noodles_Step, output_dims ## input_dims
    assert len(dout3.attrs) == 2   # Check step/stop are parsed as dimensions    
    
if __name__ == "__main__":
    test_datwrapper_basic()
    test_datwrapper_warnings()
    test_datwrapper_create_data_array()
