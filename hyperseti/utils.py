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

