import os
import numpy as np
from astropy.coordinates import SkyCoord
# Filterbank reading
from blimpy.io import sigproc
from hyperseti.dimension_scale import TimeScale, DimensionScale
from hyperseti.data_array import DataArray

def from_fil(filename):
    """ Create a DataArray from a sigproc filterbank file
    
    Args:
        filename (str): Path to filterbank file.
    
    Returns a DataArray object with mem-mapped filterbank data.
    """
    hdr    = sigproc.read_header(filename)
    hdrlen = sigproc.len_header(filename)
    n_int  = sigproc.calc_n_ints_in_file(filename)
    shape  = (n_int,  hdr['nbeams'], hdr['nchans'])
    data   = np.memmap(filename=filename, dtype='float32', offset=hdrlen, shape=shape)
    
    attrs = {'name': os.path.basename(filename),
             'source': hdr['source_name'],
             'sky_coord':  SkyCoord(hdr['src_raj'], hdr['src_dej'])}
    
    dims  = ('time', 'beam_id', 'frequency')
    scales = {
        'time':      TimeScale('time', hdr['tstart'], hdr['tsamp'], data.shape[0], time_format='mjd', time_delta_format='sec'),
        'beam_id':   DimensionScale('beam_id', 0, 0, data.shape[1], units=''),
        'frequency': DimensionScale('frequency', hdr['fch1'], hdr['foff'], data.shape[2], units='MHz')
    }
    
    d = DataArray(data, dims, scales, attrs, units='counts')
    return d