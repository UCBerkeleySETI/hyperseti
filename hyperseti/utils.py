import cupy as cp
import numpy as np
from functools import wraps
from inspect import signature
from astropy.units import Unit, Quantity
import numpy as np
import cupy as cp
import pandas as pd
import warnings
import copy

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
    func_params = list(signature(func).parameters.keys())
    @wraps(func)
    def inner(*args, **kwargs):
        new_args = []
        logger.debug(f"{func_name} on_gpu inner, args: {args}")
        for idx, arg in enumerate(args):
            argname = func_params[idx]
            if isinstance(arg, np.ndarray):
                logger.info(f"<{func_name}> Converting ndarray arg {argname} to cupy..")
                if arg.dtype != np.dtype('float32'):
                    logger.warning(f"<{func_name}> Arg {argname} is not float32, could cause issues...")
                arg = cp.asarray(arg)
            elif hasattr(arg, '__array__'):
                # Duck-type numpy array
                logger.info(f"<{func_name}> Converting numpy-like arg {argname} to cupy..")
                if arg.dtype != np.dtype('float32'):
                    warnings.warn(f"<{func_name}> Arg {argname} is not float32, could cause issues...", RuntimeWarning)
                arg = cp.asarray(arg)                
            if isinstance(arg, DataArray):
                logger.info(f"<{func_name}> Converting arg {argname}.data to cupy..")
                if arg.data.dtype != np.dtype('float32'):
                    warnings.warn(f"<{func_name}> Arg {argname}.data is not float32, could cause issues...", RuntimeWarning)
                arg.data = cp.asarray(arg.data)                
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
    
    if wrapped function supplies 'output_dims' then these will be used.
    
    Notes:
        Specific for hyperseti, this will also generate frequency_step 
        and time_step from dimension scales.
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
                metadata['input_dims']  = d.dims # NB: also only intended to be used by the wrapper
                                                 # But we do supply this to the function just in case
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
                    metadata = func_params[md_idx]
                    #print(func_params)
                    #raise RuntimeError(f"<{func_name}> wrapped function does not have a 'metadata' parameter")
                    
            # OUTPUT MODIFYING
            output = func(*args, **kwargs)
            # First, check if the function returns nothing, a bare numpy/cupy array, or pandas array
            if output is None:
                return output
            elif isinstance(output, (np.ndarray, cp.core.core.ndarray)):
                if dims is not None:
                    warnings.warn(f"<{func_name}> dimensions supplied, but function returns bare numpy array (no metadata).", RuntimeWarning)
                return output
            elif isinstance(output, pd.DataFrame):
                if dims is not None:
                    warnings.warn(f"<{func_name}> dimensions supplied, but function returns pandas Dataframe (no metadata).", RuntimeWarning)
                return output                
            elif isinstance(output, (list, tuple)):
                # Check to see if we can apply original dimensions
                # This requires input to be a DataArray, and output to have array + metadata
                _dims = None
                if dims is None:
                    if len(output) >= 2:
                        has_array = isinstance(output[0], (np.ndarray, cp.core.core.ndarray))
                        has_metadata = isinstance(output[1], dict)
                        if has_array and has_metadata and isinstance(args[0], DataArray):
                            # In case the wrapped function manually added output_dims
                            if 'output_dims' in output[1].keys():
                                _dims = output[1]['output_dims']
                            # Otherwise, let's use the dimensions of the input DataArray
                            else:
                                _dims = args[0].dims
                else:
                    _dims = dims          
                # Now, if dims were found, let's use those to generate a DataArray
                if _dims is not None:
                    logger.debug(f"<{func_name}> Generating DataArray from function output, {_dims}")
                    new_output = []
                    new_data = output[0]
                    new_md   = output[1]
                    new_md['output_dims'] = _dims  # Add output_dims to metadata out
                    
                    # Get rid of input dims that aren't used anymore
                    for d in new_md.pop('input_dims', ('unset',) ):
                        #print(d, new_md['output_dims'])
                        if d not in new_md['output_dims']:
                            logger.debug(f"<{func_name}> deleting missing output dimension: {d}")
                            new_md.pop(f"{d}_start", 0)
                            new_md.pop(f"{d}_step", 0)
                    # TODO: Should we delete any keys ending with _step or _start?
                    # TODO: Check for {x}_step (for key in new_md: if key.endswith(_step): pop)

                    # Create DataArray attribute dict, which will not include start/stop scales or dims
                    array_md = copy.deepcopy(new_md)
                    array_md.pop('input_dims', 0)
                    array_md.pop('output_dims', 0)
                    logger.debug(f"<{func_name}> data shape: {new_data.shape}")

                    scales = {}
                    for dim_idx, dim in enumerate(_dims):
                        nstep = new_data.shape[dim_idx]
                        if dim == 'time':
                            time_start, time_step = array_md.pop("time_start"), array_md.pop("time_step")
                            scales[dim] = TimeScale('time', time_start.value, time_step.to('s').value, 
                                               nstep, time_format=time_start.format, time_delta_format='sec')
                        else:
                            scale_start, scale_step = array_md.pop(f"{dim}_start", 0), array_md.pop(f"{dim}_step", 0)
                            logger.debug(f"{dim} {scale_start}")
                            scale_unit = None if np.isscalar(scale_start) else scale_start.unit
                            scales[dim] = DimensionScale(dim, scale_start, scale_step, 
                                                   nstep, units=scale_unit)
                    darr = DataArray(new_data, _dims, scales=scales, attrs=array_md)
                    new_output.append(darr)
                    new_output.append(new_md)
                    for op in output[2:]:
                        new_output.append(op)
                    return new_output 
                # Otherwise, we return the un-edited output
                else:
                    return output
            else:
                if dims is not None:
                    t = type(output)
                    warnings.warn(f"<{func_name}> dimensions supplied, but function returns {t} which can't be a DataArray.", RuntimeWarning)
                return output   
        return inner
    return _datwrapper
