import cupy as cp
import numpy as np
import time
import pandas as pd

from astropy import units as u

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

def blank_hit(data_array, f0, drate, padding=4):
    """ Blank a hit in an array by setting its value to zero
    
    Args:
        data (cp.array): Data array
        metadata (dict): Metadata with frequency, time info
        f0 (astropy.Quantity): Frequency at t=0
        drate (astropy.Quantity): Drift rate to blank
        padding (int): number of channels to blank either side. 
    
    Returns:
        data (cp.array): blanked data array
    
    TODO: Add check if drate * time_step > padding
    """
    n_time, n_pol, n_chans = data_array.data.shape
    f_start = data_array.frequency.start.to('Hz').value
    f_step  = data_array.frequency.step.to('Hz').value
    t_step  = data_array.time.step.to('s').value

    i0     = int((f0 - f_start) / f_step)
    i_step =  t_step * drate / f_step
    i_off  = (i_step * cp.arange(n_time) + i0).astype('int64')
    
    min_padding = int(abs(i_step) + 1)  # i_step == frequency smearing
    padding += min_padding 
    i_time = cp.arange(n_time, dtype='int64')
    for p_off in range(padding):
        data_array.data[i_time, :, i_off] = 0
        data_array.data[i_time, :, i_off - p_off] = 0
        data_array.data[i_time, :, i_off + p_off] = 0
    return data_array


def blank_hits(data_array, df_hits, padding=4):
    """ Blank all hits in a data_array 

    Calls blank_hit() iteratively

    Args:
        data_array (DataArray): data array to apply blanking to
        df_hits (pd.DataFrame): pandas dataframe of hits to blank
    """
    for idx, row in df_hits.iterrows():
        f0, drate = float(row['f_start']), float(row['drift_rate'])
        box_width = int(row['boxcar_size'])
        data_array = blank_hit(data_array, f0, drate, padding=padding+box_width)
    return data_array