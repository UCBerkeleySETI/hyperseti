import os
import numpy as np
from astropy.coordinates import SkyCoord

from hyperseti.dimension_scale import TimeScale, DimensionScale
from hyperseti.data_array import DataArray

# HDF5 reading
import hdf5plugin
import h5py

def from_h5(filename):
    """ Create a DataArray from a HDF5 file 
    
    Args:
        filename (str): Path to h5 file (in blimpy filterbank format)
    
    Returns a DataArray object with h5py mapped data.
    """

    h5 = h5py.File(filename, mode='r')
    data  = h5['data']
    hdr = data.attrs
    attrs = {'name': os.path.basename(filename),
            'source': hdr['source_name'],
             'sky_coord':  SkyCoord(hdr['src_raj'], hdr['src_dej'], unit=('hourangle', 'deg'))}
    dims  = ('time', 'feed_id', 'frequency')
    scales = {
        'time':      TimeScale('time', hdr['tstart'], hdr['tsamp'], data.shape[0], time_format='mjd', time_delta_format='sec'),
        'feed_id':   DimensionScale('feed_id', 0, 0, data.shape[1], units=''),
        'frequency': DimensionScale('frequency', hdr['fch1'], hdr['foff'], data.shape[2], units='MHz')
    }
    
    d = DataArray(data, dims, scales, attrs, units='counts')
    return d