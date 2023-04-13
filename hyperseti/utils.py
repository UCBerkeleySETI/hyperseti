import cupy as cp
import numpy as np
from functools import wraps
from inspect import signature
from astropy.units import Unit, Quantity
import numpy as np
import cupy as cp
import pandas as pd
import copy
import time
from typing import Any, Callable

from .data_array import DataArray, from_metadata, split_metadata
from .dimension_scale import DimensionScale, TimeScale

current_gpu_id = 0

# Logging
from .log import get_logger
logger      = get_logger('hyperseti.utils')
time_logger = get_logger('hyperseti.timer')

def attach_gpu_device(new_id: int):
    """ On demand, switch to GPU ID new_id.

    Args:
        new_id (int): Integer ID of GPU to bind to
    """
    global current_gpu_id
    try:
        if new_id == current_gpu_id:
            logger.info(f"attach_gpu_device: Already using GPU ({new_id})")
        else: #pragma: no cover
              # (can't run unit test on single-GPU systems)
            cp.cuda.Device(new_id).use()
            logger.info(f"attach_gpu_device: Using device ID ({new_id})")
            current_gpu_id = new_id
    except: #pragma: no cover
        cur_id = cp.cuda.Device().id
        logger.error("attach_gpu_device: attach_gpu_device: cp.cuda.Device({}).use() FAILED!".format(new_id))
        logger.warning("attach_gpu_device: Will continue to use current device ID ({})".format(cur_id))
        raise


def timeme(func: Callable[[], Any]) -> Callable[[], Any]:
    """ Timing decorator 
    
    Usage:
        from log import set_log_level
        set_log_level('hyperseti.timer', 'info')

        @timeme
        def do_something(x):
            return x
    """
    def wrapper(*arg, **kwargs):
        t1 = time.time()
        res = func(*arg, **kwargs)
        tt = time.time() - t1
        time_logger.info(f"Time taken: {func.__name__}, {tt:.3f} s")
        return res
    return wrapper