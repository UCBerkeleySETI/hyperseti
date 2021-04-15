from .dimension_scale import DimensionScale, TimeScale
from blimpy.io import sigproc
import numpy as np
import h5py
from astropy.coordinates import SkyCoord
import itertools

def iterate_through_dataset(ds, dims, overlap={}):
    """ modified from https://github.com/pangeo-data/xbatcher """
    def _slices(dimsize, size, overlap=0):
        """ return a list of slices to chop up a single dimension """
        slices = []
        print(dimsize, size, overlap)
        stride = size - overlap
        assert stride > 0
        assert stride < dimsize
        for start in range(0, dimsize, stride):
            end = start + size
            if end <= dimsize:
                slices.append(slice(start, end))
        return slices
    dim_slices = []
    for dim in dims:
        print(dim)
        dim_idx = ds.dims.index(dim)
        print(dim_idx)
        dimsize = ds.data.shape[dim_idx]
        size = dims[dim]
        olap = overlap.get(dim, 0)
        dim_slices.append(_slices(dimsize, size, olap))

    for slices in itertools.product(*dim_slices):
        selector = {key: slice for key, slice in zip(dims, slices)}
        yield selector

class DataArray(object):
    def __init__(self, data, dims, scales, attrs):
        
        # Check correct setup
        assert isinstance(attrs, dict)
        assert len(data.shape) == len(dims)
        assert isinstance(scales, dict)
        for key in scales.keys():
            try:
                assert key in dims
            except:
                raise ValueError(f"Key {key} not in {dims}")
                
        self.attrs = attrs
        self.data   = data
        self.dims   = dims
        self.scales = scales
        self.chunks = None
    
    def isel(self, sel):
        """ Select by index """
        slices = []
        new_scales = {}
        for idx, dim_id in enumerate(self.dims):
            print(idx, dim_id)
            sl = sel.get(dim_id, slice(None))
            slices.append(sl)
            new_scales[dim_id] = self.scales[dim_id][sl]
        slices = tuple(slices)
        data = self.data[slices]
        return DataArray(data, self.dims, new_scales, self.attrs)

    def iterate_through_data(self, dims, overlap={}):
        """ modified from https://github.com/pangeo-data/xbatcher """
        def _slices(dimsize, size, overlap=0):
            """ return a list of slices to chop up a single dimension """
            slices = []
            print(dimsize, size, overlap)
            stride = size - overlap
            assert stride > 0
            assert stride < dimsize
            for start in range(0, dimsize, stride):
                end = start + size
                if end <= dimsize:
                    slices.append(slice(start, end))
            return slices
        dim_slices = []
        for dim in dims:
            print(dim)
            dim_idx = self.dims.index(dim)
            print(dim_idx)
            dimsize = self.data.shape[dim_idx]
            size = dims[dim]
            olap = overlap.get(dim, 0)
            dim_slices.append(_slices(dimsize, size, olap))

        for slices in itertools.product(*dim_slices):
            selector = {key: slice for key, slice in zip(dims, slices)}
            yield self.isel(selector)

def from_fil(filename):
    hdr    = sigproc.read_header(filename)
    hdrlen = sigproc.len_header(filename)
    n_int  = sigproc.calc_n_ints_in_file(filename)
    shape  = (n_int,  hdr['nbeams'], hdr['nchans'])
    data = np.memmap(filename=filename, dtype='float32', offset=hdrlen, shape=shape)
    
    attrs = {'name': filename,
             'source': hdr['source_name'],
             'sky_coord':  SkyCoord(hdr['src_raj'], hdr['src_dej'])}
    
    dims  = ('time', 'feed_id', 'frequency')
    scales = {
        'time':      TimeScale('time', hdr['tstart'], hdr['tsamp'], data.shape[0], time_format='mjd', time_delta_format='sec'),
        'feed_id':   DimensionScale('feed_id', 0, 0, data.shape[1], units=''),
        'frequency': DimensionScale('frequency', hdr['fch1'], hdr['foff'], data.shape[2], units='MHz')
    }
    
    d = DataArray(data, dims, scales, attrs)
    return d

def from_h5(filename):
    h5 = h5py.File(filename, mode='r')
    data  = h5['data']
    hdr = data.attrs
    attrs = {'name': filename,
            'source': hdr['source_name'],
             'sky_coord':  SkyCoord(hdr['src_raj'], hdr['src_dej'], unit=('hourangle', 'deg'))}
    dims  = ('time', 'feed_id', 'frequency')
    scales = {
        'time':      TimeScale('time', hdr['tstart'], hdr['tsamp'], data.shape[0], time_format='mjd', time_delta_format='sec'),
        'feed_id':   DimensionScale('feed_id', 0, 0, data.shape[1], units=''),
        'frequency': DimensionScale('frequency', hdr['fch1'], hdr['foff'], data.shape[2], units='MHz')
    }
    
    d = DataArray(data, dims, scales, attrs)
    return d
    
    