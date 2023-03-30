import cupy as cp
import numpy as np
import time
import pandas as pd

from astropy import units as u

from .data_array import DataArray
from .kernels.blank_hits import blank_hits_kernel

#logging
from .log import get_logger
logger = get_logger('hyperseti.blanking')

def blank_edges(data_array: DataArray, n_chan: int) -> DataArray:
    """ Blank n_chan at edge of data array 
    Args:
        data_array (DataArray): Data array to blank extrema in 
        n_chan (int): Number of channels to blank on left and right side
    
    Returns:
        data (DataArray): blanked data array
    """
    data_array.data[..., :n_chan] = 0
    data_array.data[..., data_array.data.shape[-1] - n_chan:] = 0
    return data_array


def blank_extrema(data_array: DataArray, threshold: float, do_neg: bool=False) -> DataArray:
    """ Blank really bright things 
    
    Args:
        data_array (DataArray): Data array to blank extrema in 
        threshold (float): Values above which to blank as 'extrema'
        do_neg (bool): Blank less than negative threshold
    Returns:
        data (DataArray): blanked data array
    """
    to_blank = data_array.data > threshold
    data_array.data[to_blank] = 0
    if do_neg:
        to_blank = data_array.data < -threshold
        data_array.data[to_blank] = 0
    return data_array


def blank_hit(data_array: DataArray, f0: u.Quantity, drate: u.Quantity, padding: int=4) -> DataArray:
    """ Blank a hit in an array by setting its value to zero
    
    Args:
        data (DataArray): Data array (time. beam_id, frequency)
        metadata (dict): Metadata with frequency, time info
        f0 (astropy.Quantity): Frequency at t=0
        drate (astropy.Quantity): Drift rate to blank
        padding (int): number of channels to blank either side. 
    
    Returns:
        data (DataArray): blanked data array
    
    TODO: Add check if drate * time_step > padding
    """
    n_time, n_pol, n_chans = data_array.data.shape

    if isinstance(drate, u.Quantity):
        drate = drate.to('Hz/s').value
 
    f_step  = data_array.frequency.step.to('Hz').value
    t_step  = data_array.time.step.to('s').value

    i0 = data_array.frequency.index(f0)
    i_step = t_step * drate / f_step

    i_off  = (i_step * cp.arange(n_time) + i0).astype('int64')
    
    min_padding = int(abs(i_step) + 1)  # i_step == frequency smearing
    padding += min_padding 
    i_time = cp.arange(n_time, dtype='int64')
    for p_off in range(padding):
        data_array.data[i_time, :, i_off] = 0
        data_array.data[i_time, :, i_off - p_off] = 0
        data_array.data[i_time, :, i_off + p_off] = 0
    return data_array


def blank_hits(data_array: DataArray, df_hits: pd.DataFrame, padding: int=4) -> DataArray:
    """ Blank all hits in a data_array 
    Calls blank_hit() iteratively
    Args:
        data_array (DataArray): data array to apply blanking to
        df_hits (pd.DataFrame): pandas dataframe of hits to blank
    
    Returns:
        data_array (DataArray): Blanked data array
    """
    for idx, row in df_hits.iterrows():
        f0, drate = float(row['f_start']), float(row['drift_rate'])
        box_width = int(row['boxcar_size'])
        data_array = blank_hit(data_array, f0, drate, padding=padding+box_width)
    return data_array

def blank_hits_gpu(data_array: DataArray, df_hits: pd.DataFrame, padding: int=4) -> DataArray:
    """ Blank all hits in a data_array 
    Calls blank_hit() iteratively
    Args:
        data_array (DataArray): data array to apply blanking to
        df_hits (pd.DataFrame): pandas dataframe of hits to blank
    
    Returns:
        data_array (DataArray): Blanked data array
    
    TODO: Get this to work with mutiple polarizations.
    """
    
    # Setup grid and block dimensions
    N_blank   = len(df_hits)
    N_threads = np.min((N_blank, 1024))
    N_grid    = N_blank // N_threads
    if N_blank % N_threads != 0:
        N_grid += 1
    logger.debug(f"blank_hits: Kernel shape (grid, block) {(N_grid, ), (N_threads,)}")

    d_gpu = data_array.data
    N_chan, N_pol, N_time = d_gpu.shape
    cidxs_gpu = cp.asarray(df_hits['channel_idx'], dtype='int32')

    boxcar_size_gpu = cp.asarray(df_hits['boxcar_size'], dtype='int32')

    # Convert dedoppler Hz/s into channels/timestep (can't use driftrate_idx in case they used 'stepped')
    df, dt = data_array.frequency.step.to('Hz').value, data_array.time.step.to('s').value
    dd_shift_gpu = cp.asarray(np.round(df_hits['drift_rate'] / (df / dt)), dtype='int32')

    blank_hits_kernel((N_grid, 1, 1), (N_threads, 1, 1), 
                      (d_gpu, cidxs_gpu, dd_shift_gpu, boxcar_size_gpu, padding, N_chan, N_pol, N_time, N_blank)) 

    data_array.data = d_gpu
    return data_array