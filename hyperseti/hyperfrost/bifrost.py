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

def gen_bf_metadata(data_array: DataArray) -> dict:
    """ Generate bifrost metadata dict from DataArray """
    metadata = deepcopy(data_array.attrs)

    for k, v in metadata.items():
        if isinstance(v, SkyCoord):
            metadata[k] = (v.ra.value, v.dec.value)
    
    scales, units = [], []
    for ii, dim_id in enumerate(data_array.dims):
        ds = data_array.scales[dim_id]
        scales.append([ds.start.value, ds.step.value])
        units.append(ds.units.to_string())

        # Bifrost reserves the keyword 'time'
        #if dim_id == 'time':
        #    dimlist = list(data_array.dims)
        #    dimlist[ii] = 'frame_time'
        #    data_array.dims = tuple(dimlist)
        #    ds = data_array.scales.pop(dim_id)
        #    data_array.scales['frame_time'] = ds

    tensor = {
        'dtype': numpy2string(np.dtype(data_array.data.dtype)),
        'shape': [-1] + list(data_array.shape),
        'labels': ['frame'] + list(data_array.dims),
        'units' : ['s'] + list(units),
        'scales': [[0, 1]] + list(scales)
    }

    metadata['_tensor'] = tensor
    return metadata

def to_bf(data_array: DataArray) -> (bf.ndarray, dict):
    """ Convert data array object to bifrost ndarray + metadata 
    
    Args:
        data_array (DataArray): Data array to convert
    
    Returns:
        bf_array (bf.ndarray): Bifrost ndarray
        metadata (dict): Dictionary of metadata
    """

    d_bf = bf.ndarray(data_array.data)
    metadata = gen_bf_metadata(data_array)

    return d_bf, metadata
