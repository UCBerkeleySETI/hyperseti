import cupy as cp
import numpy as np
from functools import wraps
from inspect import signature
from astropy.units import Unit

from .data_array import DataArray
from .dimension_scale import DimensionScale, TimeScale

# Logging
from .log import logger_group, Logger
logger = Logger('hyperseti.utils')
logger_group.add_logger(logger)

def on_gpu(func):
    """ Decorator to automatically copy a numpy array over to cupy.
    
    Checks if input data array is numpy.ndarray and converts it to 
    cupy.ndarray if it is.
    
    Also adds a 'return_space' kwarg to the decorated funciton, 
    which returns data in either 'cpu' (numpy) or 'gpu' (cupy) space.
    
    Usage example:
        @on_gpu
        def compute_something(x):
            return cp.sum(x)**2 / 1.234
        
        x = np.arange(0, 1024, dtype='float32')
        compute_something(x, return_space='cpu')
        
    """
    func_name = func.__name__
    @wraps(func)
    def inner(*args, **kwargs):
        new_args = []
        for idx, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                logger.info(f"<{func_name}> Converting arg {idx} to cupy..")
                if arg.dtype != np.dtype('float32'):
                    logger.warning(f"<{func_name}> Arg {idx} is not float32, could cause issues...")
                arg = cp.asarray(arg)
            new_args.append(arg)
            
        return_space = None
        if 'return_space' in kwargs:
            logger.debug(f"<{func_name}> Return space requested: {kwargs['return_space']}")
            return_space = kwargs.pop('return_space')
            assert return_space in ('cpu', 'gpu')
        output = func(*new_args, **kwargs)
        
        if return_space == 'gpu':
            if len(output) == 1 or isinstance(output, (np.ndarray, cp.ndarray)):
                if isinstance(output, np.ndarray):
                    logger.info(f"<{func_name}> Converting output to cupy")
                    output = cp.asarray(output)
                return output
            else:
                new_output = []
                for idx, item in enumerate(output):
                    if isinstance(item, np.ndarray):
                        logger.info(f"<{func_name}> Converting output {idx} to cupy")
                        item = cp.asarray(item)
                    new_output.append(item)
                return new_output
            
        elif return_space == 'cpu':
            if len(output) == 1 or isinstance(output, (np.ndarray, cp.ndarray)):
                if isinstance(output, cp.ndarray):
                    logger.info(f"<{func_name}> Converting output to numpy")
                    output = cp.asnumpy(output)
                return output
            else:
                new_output = []
                for idx, item in enumerate(output):
                    if isinstance(item, cp.ndarray):
                        logger.info(f"<{func_name}> Converting output {idx} to numpy")
                        item = cp.asnumpy(item)
                    new_output.append(item)
                return new_output
        else:
            return output 
    return inner

            
def datwrapper(dims=None, *args, **kwargs):
    """ Decorator to split metadata off from DataArray 
    
    Supplies metadata= kwarg to wrapped function, derived from
    attributes of the DataArray. Splits off the DataArray.data
    and returns that as first argument.
    
    wrapped funcion must have data as first output, and metadata dict
    
    Notes:
        Specific for hyperseti, this will also generate df and dt from
        dimension scales. 
    """
    def _datwrapper(func, *args, **kwargs):
        func_name = func.__name__
        @wraps(func)
        def inner(*args, **kwargs):
            # INPUT MODIFYING
            if isinstance(args[0], DataArray):
                args = list(args)
                d = args[0]
                metadata = {}
                # Copy attribute key:values over 
                for k, v in d.attrs.items():
                    metadata[k] = v
                metadata['dims'] = dims
                for dim in d.dims:
                    if dim == 'time':
                        metadata['time_start']  = d.time.time_start
                        metadata['time_step'] = d.time.units * d.time.val_step
                    else:
                        scale = d.scales[dim]
                        scale.units = Unit('') if scale.units is None else scale.units
                        logger.debug(f"{dim} {scale}")
                        metadata[f"{dim}_start"] = scale.units * scale.val_start
                        metadata[f"{dim}_step"]  = scale.units * scale.val_step

                # Replace DataArray with underlying data
                args[0] = d.data

                # Check if metadata is an argument of the function to be called
                if 'metadata' in signature(func).parameters:
                    kwargs['metadata'] = metadata
            else:
                try:
                    metadata = kwargs['metadata']
                except KeyError:
                    # Try and get metadata by looking up argument and finding index
                    func_params = list(signature(func).parameters.keys())
                    md_idx = func_params.index('metadata')
                        
            # OUTPUT MODIFYING
            output = func(*args, **kwargs)
            if dims is not None:
                logger.debug(f"<{func_name}> Generating DataArray from function output, {dims}")
                new_output = []
                new_data = output[0]
                new_md   = output[1]
                logger.debug(f"<{func_name}> data shape: {new_data.shape}")

                scales = {}
                for dim_idx, dim in enumerate(dims):
                    nstep = new_data.shape[dim_idx]
                    if dim == 'time':
                        time_start, time_step = new_md["time_start"], new_md["time_step"]
                        scales[dim] = TimeScale('time', time_start.value, time_step.to('s').value, 
                                           nstep, time_format=time_start.format, time_delta_format='sec')
                    else:
                        scale_start, scale_step = new_md.get(f"{dim}_start", 0), new_md.get(f"{dim}_step", 0)
                        logger.debug(f"{dim} {scale_start}")
                        scale_unit = None if np.isscalar(scale_start) else scale_start.unit
                        scales[dim] = DimensionScale(dim, scale_start, scale_step, 
                                               nstep, units=scale_unit)
                darr = DataArray(new_data, dims, scales=scales, attrs=new_md)
                new_output.append(darr)
                new_output.append(new_md)
                for op in output[2:]:
                    new_output.append(op)
                return new_output 
            else:
                return output      
        return inner
    return _datwrapper
