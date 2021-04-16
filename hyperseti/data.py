import numpy as np
import os
from astropy.coordinates import SkyCoord
import itertools

# Filterbank reading
from blimpy.io import sigproc

# HDF5 reading
import hdf5plugin
import h5py

# Dask SVG HTML plotting
from dask.array.svg import svg

# hyperseti
from .dimension_scale import DimensionScale, TimeScale

# Logging
from .log import logger_group, Logger
logger = Logger('hyperseti.data')
logger_group.add_logger(logger)

class DataArray(object):
    """ An N-dimensional array with dimension scales and metadata

    Designed for large out-of-memory data to read via iterative chunks.
    Provides an isel() 
    
    Provides two main methods:
       * isel() - method to read slices into memory, and compute
         corresponding dimension scale values. Returns a new DataArray
       * iterate_through_data() - method to easily iterate through chunks
         of data. This is a generator which yields DataArrays
    
    End users of hyperseti should use from_fil() or from_h5() and should
    not call this directly.
    
    Constructor must have:
        data: np.array like (e.g. h5py dataset, np.memmap dataset)
        dims: tuple of dimension labels, e.g. (time, frequency)
        scales: dict of {dim: DimensionScale}
        attrs: dict of metadata attributes (can be anything)
    """
    def __init__(self, data, dims, scales, attrs, slice_info=None):
        
        # Check correct setup
        assert isinstance(attrs, dict)
        assert len(data.shape) == len(dims)
        assert isinstance(scales, dict)
        for key in scales.keys():
            try:
                assert key in dims
            except:
                raise ValueError(f"Key {key} not in {dims}")
                
        self.attrs  = attrs
        self.data   = data
        self.dims   = dims
        self.scales = scales
        self.slice_info = slice_info
        
        for dim in dims:
            self.__dict__[dim] = self.scales[dim]
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __repr__(self):
        r = f"<DataArray: shape={self.shape}, dims={self.dims}>"
        return r
    
    def _repr_html_(self):
        
        chunks = [(s,) for s in self.data.shape]
        arr_img = svg(chunks)
        
        table = [
            "<table>",
            "  <thead>",
            "    <tr><td> </td><th> Array </th></tr>",
            "  </thead>",
            "  <tbody>",
            f"    <tr><th> Shape </th><td> {self.data.shape} </td> </tr>",
            f"    <tr><th> Dims </th><td> {self.dims} </td> </tr>",
            "  </tbody>",
            "</table>",
        ]
        
        attrs_table = ["<table>", 
                       "<thead><tr><th></th><th>Attributes</th></tr></thead>",
                      "<tbody>"]
        for k, v in self.attrs.items():
            attrs_table.append(f"<tr><th>{k}</th> <td>{str(v).strip('<>')}</td> </tr>")
        attrs_table.append("</tbody></table>")
        
        dims_table = ["<table>", "<thead><tr> <td></td><th>Dimension Scales</th></tr></thead>"]
        for k, v in self.scales.items():
            dims_table.append(f"<tr><th>{k}</th> <td>{v.val_start, v.val_step}</td> </tr>")
        dims_table.append("</tbody></table>")
        
        table = "\n".join(table)
        
        html = [
            "<table>",
            "<tr>",
            "<td>",
            table,
            *attrs_table,
            *dims_table,
            "</td>",
            "<td>",
            arr_img,
            "</td>",
            "</tr>",
            "</table>",
        ]
        return "\n".join(html)
    
    def isel(self, sel):
        """ Select subset of data using slices along specified dimension.
        
        Args:
            sel (dict): Dictionary of {dim: slice(start, stop, stride)}.
        
        Example usage:
            # Select every second sample along freq axis, from 0-1000
            ds.isel({'frequency': slice(0, 1000, 2)})   
            # select first 4096 freq channels for first 10 timesteps
            d_h5.isel({'frequency': slice(0, 4096), 'time': slice(0, 10)})
            
        """
        slices = []
        new_scales = {}
        for idx, dim_id in enumerate(self.dims):
            sl = sel.get(dim_id, slice(None))
            slices.append(sl)
            new_scales[dim_id] = self.scales[dim_id][sl]
        slices = tuple(slices)
        data = self.data[slices]
        logger.debug(f"isel data shape: {data.shape}")
        return DataArray(data, self.dims, new_scales, self.attrs, slice_info=slices)

    def iterate_through_data(self, dims, overlap={}):
        """ Generator to iterate through chunks of data
        
        Args:
            dims (dict): Dictionary of {dim: chunk_size}.
        
        Example usage:
            # iterate over blocks of frequency and time
            for ds_sub in ds.iterate_through_data({'frequency': 1024, 'time': 10}):
                print(ds_sub.shape)
        
        Notes:
            modified from https://github.com/pangeo-data/xbatcher 
        """
        def _slices(dimsize, size, overlap=0):
            """ return a list of slices to chop up a single dimension """
            slices = []
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
            dim_idx = self.dims.index(dim)
            dimsize = self.data.shape[dim_idx]
            size = dims[dim]
            if size == dimsize:
                dim_slices.append([slice(0, dimsize, None)]) # Read whole dim
            else:
                olap = overlap.get(dim, 0)
                dim_slices.append(_slices(dimsize, size, olap))
        
        logger.debug(f"dim_slices len({len(dim_slices)}), {dim_slices}")
        
        for slices in itertools.product(*dim_slices):
            selector = {key: slice for key, slice in zip(dims, slices)}
            yield self.isel(selector)

            
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
    
    dims  = ('time', 'feed_id', 'frequency')
    scales = {
        'time':      TimeScale('time', hdr['tstart'], hdr['tsamp'], data.shape[0], time_format='mjd', time_delta_format='sec'),
        'feed_id':   DimensionScale('feed_id', 0, 0, data.shape[1], units=''),
        'frequency': DimensionScale('frequency', hdr['fch1'], hdr['foff'], data.shape[2], units='MHz')
    }
    
    d = DataArray(data, dims, scales, attrs)
    return d


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
    
    d = DataArray(data, dims, scales, attrs)
    return d
    
