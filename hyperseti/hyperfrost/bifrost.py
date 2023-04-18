import os
import numpy as np
from astropy.coordinates import SkyCoord

from hyperseti.dimension_scale import TimeScale, DimensionScale
from hyperseti.data_array import DataArray

import bifrost as bf
from bifrost.dtype import numpy2string
from copy import deepcopy

# HDF5 reading
import hdf5plugin
import h5py

def from_bf(bf_array: bf.ndarray, metadata: dict) -> DataArray:
    """ Create a DataArray from a bifrost object + metadata
    
    Args:
        bf_array (bifrost.ndarray): Bifrost ndarray
        metadata (dict): Bifrost metadata dictionary
    
    Returns a DataArray object with h5py mapped data.
    
    Notes:
    Bifrost tensor has form:
        ohdr = {
            '_tensor': {
                'dtype':  ['u', 'i'][ihdr['signed']] + str(nbit),
                'shape':  [-1, ihdr['nifs'], ihdr['nchans']],
                'labels': ['time', 'pol', 'freq'],
                'scales': [(tstart_unix, ihdr['tsamp']),  None, (ihdr['fch1'], ihdr['foff'])],
                'units':  ['s', None, 'MHz']
            }
    """

    data = bf_array.as_cupy()

    attrs = deepcopy(metadata)

    tensor = attrs.pop('_tensor')
    dims  = tensor['labels']
    
    scales = {}
    for ii, dim_id in enumerate(dims):
        try:
            start, step = tensor['scales'][ii]
            if start is None: start = 0
            if step is None: step = 1
        except TypeError:
            start, step = 0, 0

        unit = tensor['units'][ii]
        if unit is None: unit = ''

        assert list(data.shape) == list(tensor['shape'])
        shape = data.shape

        scales[dim_id] = DimensionScale(dim_id, start, step, shape, units=unit)

    d = DataArray(data, dims, scales, attrs, units='')
    return d

def to_bf(data_array: DataArray) -> (bf.ndarray, dict):
    """ Convert data array object to bifrost ndarray + metadata 
    
    Args:
        data_array (DataArray): Data array to convert
    
    Returns:
        bf_array (bf.ndarray): Bifrost ndarray
        metadata (dict): Dictionary of metadata
    """

    d_bf = bf.ndarray(data_array.data)
    metadata = deepcopy(data_array.attrs)
    
    scales, units = [], []
    for dim_id in data_array.dims:
        ds = data_array.scales[dim_id]
        scales.append([ds.start.value, ds.step.value])
        units.append(ds.units.to_string())

    tensor = {
        'dtype': numpy2string(np.dtype(str(d_bf.dtype))),
        'shape': d_bf.shape,
        'labels': data_array.dims,
        'units' : units,
        'scales': scales
    }

    metadata['_tensor'] = tensor
    return d_bf, metadata
