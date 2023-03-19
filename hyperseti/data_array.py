import numpy as np
import os
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
import itertools
import cupy as cp
from copy import deepcopy

# Dask SVG HTML plotting
from dask.array.svg import svg

# hyperseti
from .dimension_scale import DimensionScale, TimeScale, ArrayBasedDimensionScale

# Logging
from .log import get_logger
logger = get_logger('hyperseti.data_array')


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
    def __init__(self, data, dims, scales, attrs, units=None, slice_info=None, parent_shape=None):
        
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
        self.units  = units
        
        for dim in dims:
            self.__dict__[dim] = self.scales[dim]
        
        self.slice_info   = slice_info
        self.parent_shape = parent_shape
        self._is_slice = False if slice_info is None else True
    
    @property
    def shape(self):
        return self.data.shape

    @property
    def space(self):
        if isinstance(self.data, cp.ndarray):
            return 'gpu'
        else:
            return 'cpu'
    
    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def metadata(self):
        """ Split off metadata and return array + dict
        
        Args:
            data_array (DataArray): Data array to split
        
        Returns:
            data, metadata (array + dict)
        """
        metadata = deepcopy(self.attrs)
        metadata["dims"] = self.dims

        for scale_name, scale in self.scales.items():
            if not isinstance(scale, ArrayBasedDimensionScale):       
                metadata[f"{scale_name}_step"] = scale.val_step
                metadata[f"{scale_name}_start"] = scale.val_start
            
                if scale.units is not None:
                    metadata[f"{scale_name}_step"] *= scale.units
                    metadata[f"{scale_name}_start"] *= scale.units
        
        if self.slice_info is not None:
            metadata['slice_info'] = self.slice_info

        return metadata
        
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
            f"    <tr><th> Dimensions </th><td> {self.dims} </td> </tr>"
            f"    <tr><th> Space </th><td> {self.space} </td> </tr>"
        ]
        

                     
        if self._is_slice:
            table.append(f"    <tr><th> Slice info </th><td> {self.slice_info} </td> </tr>")  
        table.append(f"    <tr><th> Data units </th><td> {self.units} </td> </tr>")
        container_str = str(type(self.data)).split("'")[1]
        table.append(f"    <tr><th> Array type </th><td> {container_str, self.data.dtype} </td> </tr>")
        
        table += ["  </tbody>", "</table>"]
        
        attrs_table = ["<table>", 
                       "<thead><tr><th></th><th>Attributes</th></tr></thead>",
                      "<tbody>"]
        for k, v in self.attrs.items():
            attrs_table.append(f"<tr><th>{k}</th> <td>{str(v).strip('<>')}</td> </tr>")
        attrs_table.append("</tbody></table>")
        
        dims_table = ["<table>", "<thead><tr> <td></td><th>Dimension Scales (start, step) </th></tr></thead>"]
        for k, v in self.scales.items():
            v_units = '' if v.units is None else v.units
            if not isinstance(v, ArrayBasedDimensionScale): 
                dims_table.append(f"<tr><th>{k}</th> <td>{v.val_start, v.val_step} {v_units}</td> </tr>")
            else:
                 dims_table.append(f"<tr><th>{k}</th> <td>{v.data[0], 'non-uniform'} {v_units}</td> </tr>")
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
    
    def __array__(self):
        """ Returns an evaluated numpy array when np.asarray called. 
        
        See https://numpy.org/neps/nep-0030-duck-array-protocol.html
        """
        if isinstance(self.data, cp.ndarray):
            return cp.asnumpy(self.data)
        else:
            return self.data[:]

    def __duckarray__(self):
        """ Returns itself (original object) 
        
        Note: proposed in NEP 30 but not yet implemented/supported.
        Idea is to return this when np.duckarray is called.
        See https://numpy.org/neps/nep-0030-duck-array-protocol.html
        """
        return self.data    
    
    def isel(self, sel, space=None):
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
        if space == 'cpu':
            data = cp.asnumpy(data)
        elif space == 'gpu':
            data = cp.asarray(data)
        else:
            pass   
        return DataArray(data, self.dims, new_scales, self.attrs, units=self.units,
                         slice_info=slices, parent_shape=self.data.shape)
    

    def iterate_through_data(self, dims, overlap={}, space=None):
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
            yield self.isel(selector, space=space)

    def apply_transform(self, transform, *args, **kwargs):
        """ Apply a tranformation function (e.g. np.log) to the data 
        
        Args:
            transform (str or function): Transform function to apply, e.g. 'log', or np.log
                                         If a string is passed, a lookup will be done to find
                                         the corresponding numpy or cupy function.
        """
        func = None
        if callable(transform):
            func = transform
        elif isinstance(self.data, cp.ndarray):
            cp_funcs = dir(cp)
            if transform in cp_funcs:
                func = getattr(cp, transform)
        else:
            np_funcs = dir(np)
            if transform in np_funcs:
                func = getattr(np, transform)
        if func is None:
            raise RuntimeError(f"Could not interpret {transform} as cupy/numpy or callable function")
        self.data = func(self.data, *args, **kwargs)
    
    def split_metadata(self):
        return split_metadata(self)
    
    def __len__(self):
        return len(self.data)


def from_metadata(darray, metadata, dims=None, units=None):
    """ Create a data array from an array + metadata 
    
    Args:
        darray (array-like): Data array
        metadata (dict): Metadata dictionary
        dims (list): Names for dimensions of darray
    
    Returns:
        d (DataArray): Data array
    """
    scales = {}
    attrs = deepcopy(metadata)

    if dims is None:
        try:
            dims = attrs.pop("dims")
        except:
            raise RuntimeError("Dimensions must be supplied in metadata or as kwarg")

    for dim_idx, dim in enumerate(dims):
        nstep = darray.shape[dim_idx]
        if dim == 'time':
            time_start, time_step = attrs.pop("time_start"), attrs.pop("time_step")
            # Handle start time as astropy Quantity, or astropy Time
            time_format = 'unix' if isinstance(time_start, Quantity) else time_start.format
            scales[dim] = TimeScale('time', time_start.value, time_step.to('s').value, 
                                nstep, time_format=time_format, time_delta_format='sec')
        else:
            scale_start, scale_step = attrs.pop(f"{dim}_start", 0), attrs.pop(f"{dim}_step", 0)
            logger.debug(f"{dim} {scale_start}")
            scale_unit = None if np.isscalar(scale_start) else scale_start.unit
            scales[dim] = DimensionScale(dim, scale_start, scale_step, 
                                    nstep, units=scale_unit)
        
    
    d = DataArray(darray, dims, scales, attrs, units)
    return d

def split_metadata(data_array):
    """ Split off metadata and return array + dict
    
    Args:
        data_array (DataArray): Data array to split
    
    Returns:
        data, metadata (array + dict)
    """
    metadata = deepcopy(data_array.attrs)
    metadata["dims"] = data_array.dims

    for scale_name, scale in data_array.scales.items():
        if not isinstance(scale, ArrayBasedDimensionScale):       
            metadata[f"{scale_name}_step"] = scale.val_step
            metadata[f"{scale_name}_start"] = scale.val_start
        
            if scale.units is not None:
                metadata[f"{scale_name}_step"] *= scale.units
                metadata[f"{scale_name}_start"] *= scale.units
        else:
            metadata[scale_name] = scale.data
            
    if data_array.slice_info is not None:
        metadata['slice_info'] = data_array.slice_info

    return data_array.data, metadata


