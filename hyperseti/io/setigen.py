import os
import numpy as np
from astropy.coordinates import SkyCoord
import setigen as stg

from hyperseti.dimension_scale import TimeScale, DimensionScale
from hyperseti.data_array import DataArray

def from_setigen(stg_frame: stg.Frame) -> DataArray:
    """ Create a DataArray from a setigen Frame object 
    
    Args:
        stg_frame (setigen.Frame): Setigen frame object
    
    Returns a DataArray object generated from setigen.
    """

    data  = np.expand_dims(stg_frame.data, axis=1)

    attrs = {'name': 'setigen',
            'source': 'setigen data'}

    dims  = ('time', 'beam_id', 'frequency')
    scales = {
        'time':      TimeScale('time', stg_frame.ts[0], stg_frame.dt, 
                               data.shape[0], time_format='unix', time_delta_format='sec'),
        'beam_id':   DimensionScale('beam_id', 0, 0, data.shape[1], units=''),
        'frequency': DimensionScale('frequency', stg_frame.fch1, stg_frame.df, 
                                    data.shape[2], units='Hz')
    }
    
    d = DataArray(data.astype('float32'), dims, scales, attrs, units='counts')
    return d


