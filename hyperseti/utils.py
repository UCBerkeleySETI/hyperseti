import cupy as cp
import numpy as np

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
    def inner(*args, **kwargs):
        new_args = []
        for idx, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                logger.info(f"<{func_name}> Converting arg {idx} to cupy..")
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
