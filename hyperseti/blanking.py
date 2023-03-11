import cupy as cp
import numpy as np
import time
import pandas as pd
import os
import h5py

#logging
from .log import get_logger
logger = get_logger('hyperseti.blanking')

def blank_edges(data_array, n_chan):
    """ Blank n_chan at edge of data array """
    data_array.data[..., :n_chan] = 0
    data_array.data[..., data_array.data.shape[0] - n_chan:] = 0
    return data_array

def blank_extrema(data_array, threshold, do_neg=False):
    """ Blank really bright things """
    to_blank = data_array.data > threshold
    data_array.data[to_blank] = 0
    if do_neg:
        to_blank = data_array.data < -threshold
        data_array.data[to_blank] = 0
    return data_array
