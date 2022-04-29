import cupy as cp
import numpy as np
from functools import wraps
from inspect import signature
from astropy.units import Unit, Quantity
import numpy as np
import cupy as cp
import pandas as pd
import copy

from .data_array import DataArray, from_metadata, split_metadata
from .dimension_scale import DimensionScale, TimeScale

# Logging
from .log import get_logger
logger = get_logger('hyperseti.utils')


def attach_gpu_device(new_id):
    """ On demand, switch to GPU ID new_id.
    """
    try:
        cp.cuda.Device(new_id).use()
        logger.info("attach_gpu_device: Using device ID ({})".format(new_id))
    except:
        cur_id = cp.cuda.Device().id
        logger.error("attach_gpu_device: attach_gpu_device: cp.cuda.Device({}).use() FAILED!".format(new_id))
        logger.warning("attach_gpu_device: Will continue to use current device ID ({})".format(cur_id))


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

            if isinstance(arg, cp.ndarray):
                logger.debug(f"on_gpu inner <{func_name}> Arg {argname} is already a cupy array..")
            elif isinstance(arg, np.ndarray):
                logger.info(f"on_gpu inner <{func_name}> Converting ndarray arg {argname} to cupy..")
                if arg.dtype != np.dtype('float32'):
                    logger.warning(f"on_gpu inner <{func_name}> Arg {argname} is not float32, could cause issues...")
                arg = cp.asarray(arg)
            elif hasattr(arg, '__array__'):
                # Duck-type numpy array
                logger.debug(f"on_gpu inner <{func_name}> Converting numpy-like arg {argname} to cupy..")
                if arg.dtype != np.dtype('float32'):
                    logger.warning(f"<{func_name}> Arg {argname} is not float32, could cause issues...")
                arg = cp.asarray(arg)                
            if isinstance(arg, DataArray):
                if arg.data.dtype != np.dtype('float32'):
                    logger.warning(f"<{func_name}> Arg {argname}.data is not float32, could cause issues...")
                if isinstance(arg.data, cp.ndarray):
                    logger.debug(f"<{func_name}> Arg {argname}.data already cupy array..")
                else:
                    logger.info(f"<{func_name}> Converting arg {argname}.data to cupy..")
                    arg.data = cp.asarray(arg.data)                
            new_args.append(arg)
            
        return_space = None
        if 'return_space' in kwargs:
            logger.debug(f"on_gpu inner <{func_name}> Return space requested: {kwargs['return_space']}")
            return_space = kwargs.pop('return_space')
            assert return_space in ('cpu', 'gpu')
        output = func(*new_args, **kwargs)
        
        if return_space == 'gpu':
            if isinstance(output, DataArray):
                return output
            if len(output) == 1 or isinstance(output, (np.ndarray, cp.ndarray)):
                if isinstance(output, np.ndarray):
                    logger.debug(f"on_gpu inner <{func_name}> Converting output to cupy")
                    output = cp.asarray(output)
                return output
            else:
                new_output = []
                for idx, item in enumerate(output):
                    if isinstance(item, np.ndarray):
                        logger.debug(f"on_gpu inner <{func_name}> Converting output {idx} to cupy")
                        item = cp.asarray(item)
                    new_output.append(item)
                return new_output
            
        elif return_space == 'cpu':
            if len(output) == 1 or isinstance(output, (np.ndarray, cp.ndarray)):
                if isinstance(output, cp.ndarray):
                    logger.debug(f"on_gpu inner <{func_name}> Converting output to numpy")
                    output = cp.asnumpy(output)
                return output
            else:
                new_output = []
                for idx, item in enumerate(output):
                    if isinstance(item, cp.ndarray):
                        logger.debug(f"on_gpu inner <{func_name}> Converting output {idx} to numpy")
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
                data, metadata = split_metadata(args[0])
                metadata['input_dims']  = args[0].dims # NB: also only intended to be used by the wrapper
                                                      # But we do supply this to the function just in case

                # Replace DataArray with underlying data
                args[0] = data

                # Check if metadata is an argument of the function to be called
                if 'metadata' in signature(func).parameters:
                    args.insert(1, metadata)
                    #kwargs['metadata'] = metadata
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
            elif isinstance(output, DataArray):
                logger.warning(f"<{func_name}> dimensions supplied, but function natively returns DataArray. No action taken")
                return output
            elif isinstance(output, (np.ndarray, cp.ndarray)):
                if dims is not None:
                    logger.warning(f"datwrapper <{func_name}> dimensions supplied, but function returns bare numpy array (no metadata).")
                return output
            elif isinstance(output, pd.DataFrame):
                if dims is not None:
                    logger.warning(f"datwrapper <{func_name}> dimensions supplied, but function returns pandas Dataframe (no metadata).")
                return output                
            elif isinstance(output, (list, tuple)):
                # Check to see if we can apply original dimensions
                # This requires input to be a DataArray, and output to have array + metadata
                _dims = None
                if dims is None:
                    if len(output) >= 2:
                        has_array = isinstance(output[0], (np.ndarray, cp.ndarray))
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
                if _dims is not None and not isinstance(output[0], DataArray):
                    logger.debug(f"datwrapper <{func_name}> Generating DataArray from function output, {_dims}")
                    new_output = []
                    new_data = output[0]
                    
                    # Find the metadata dictionary. A bit dodgy, it assumes first dict it encounters is metadata
                    idx = 1
                    while idx < len(output):
                        if isinstance(output[idx], dict):
                            break
                        else:
                            idx += 1
                    new_md   = output[idx]
                    new_md['output_dims'] = _dims  # Add output_dims to metadata out
                    
                    # Get rid of input dims that aren't used anymore
                    for d in new_md.pop('input_dims', ('unset',) ):
                        #print(d, new_md['output_dims'])
                        if d not in new_md['output_dims']:
                            logger.debug(f"datwrapper <{func_name}> deleting missing output dimension: {d}")
                            new_md.pop(f"{d}_start", 0)
                            new_md.pop(f"{d}_step", 0)
                    # TODO: Should we delete any keys ending with _step or _start?
                    # TODO: Check for {x}_step (for key in new_md: if key.endswith(_step): pop)

                    # Create DataArray attribute dict, which will not include start/stop scales or dims
                    array_md = copy.deepcopy(new_md)
                    array_md.pop('input_dims', 0)
                    array_md.pop('output_dims', 0)
                    logger.debug(f"datwrapper <{func_name}> data shape: {new_data.shape}")
                    
                    darr = from_metadata(new_data, array_md, dims=_dims)
                    new_output.append(darr)
                    new_output.append(new_md)
                    for op in output[2:]:
                        new_output.append(op)
                    if len(new_output) == 1:
                        return new_output[0]
                    else:
                        return new_output 
                # Otherwise, we return the un-edited output
                else:
                    return output
            else:
                if dims is not None:
                    t = type(output)
                    logger.warning(f"dat_wrapper <{func_name}> dimensions supplied, but function returns {t} which can't be a DataArray.")
                return output   
        return inner
    return _datwrapper
