import cupy as cp
import numpy as np

#logging
from .log import get_logger
from .data_array import DataArray
logger  = get_logger('hyperseti.kurtosis')


def spectral_kurtosis(x: DataArray) -> cp.ndarray:
    """ GPU Generalized Spectral Kurtosis Kernel 
    
    Args:
        x (DataArray): Data to compute SK on, (frequency, beam_id, time)
    
    Returns:
        sk (cp.ndarray): Array of computed SK estimates
    
    Notes:
        Will fail if data.shape[0] == 1
    """
    metadata = x.metadata
    samps_per_sec = (1.0 / np.abs(metadata['frequency_step'])).to('s') / 2 # Nyq sample rate for channel
    N_acc = int(metadata['time_step'].to('s') / samps_per_sec)
    logger.debug(f'rescaling SK by {N_acc}')
    
    x_sum  = cp.sum(x.data, axis=0)
    x2_sum = cp.sum(x.data**2, axis=0)
    n = x.data.shape[0]
    return (N_acc*n+1) / (n-1) * (n*(x2_sum / (x_sum*x_sum)) - 1).squeeze()


def sk_flag(data_array: DataArray, n_sigma: float=None, 
            n_sigma_upper: float=10, n_sigma_lower: float=10, 
            flag_upper: bool=True, flag_lower: bool=True, 
            pad_mask: bool=True) -> cp.ndarray:
    """ Apply spectral kurtosis flagging 
    
    Args:
        data (np.array): Numpy array with shape (N_timestep, N_beam, N_channel)
        metadata (dict): Metadata dictionary, should contain 'df' and 'dt'
                         (frequency and time resolution)
        n_sigma (float): Number of std above/below to flag. (Overrides n_sigma_upper and _lower)
        n_sigma_upper (float): Number of stdev above SK estimate to flag (upper bound)
        n_sigma_lower (float): Number of stdev below SK estmate to flag (lower bound)
        flag_upper (bool): Flag channels with large SK (highly variable signals)
        flag_lower (bool): Flag channels with small SK (very stable signals)
        pad_mask (bool): Mask either side of a masked value (e.g. 00100 -> 01110)
    
    Returns:
        mask (cp.array): Array of True/False flags per channel
    
    Notes:
        sk_flag upper and lower stdev is computed on log2(sk), as the minimum
        spectral kurtosis (for a CW signal) approaches 0. 
    """
    if n_sigma is not None:
        n_sigma_lower = n_sigma
        n_sigma_upper = n_sigma

    metadata = data_array.metadata

    Fs = (1.0 / metadata['frequency_step'] / 2)
    samps_per_sec = np.abs(Fs.to('s').value) # Nyq sample rate for channel
    N_acc = int(metadata['time_step'].to('s').value / samps_per_sec)

    sk = spectral_kurtosis(data_array)

    # See SK technote 
    log_sk   = cp.log2(sk) 
    std_log  = 2.0 / np.sqrt(N_acc)         # Based on setigen
    mean_log = -1.25 / N_acc                # Based on setigen 
    
    if flag_upper and flag_lower:
        mask  = log_sk > mean_log + (std_log * n_sigma_upper)
        mask  |= log_sk < mean_log - (std_log * n_sigma_lower)
    elif flag_upper and not flag_lower:
        mask  = log_sk > mean_log + (std_log * n_sigma_upper)
    elif flag_lower and not flag_upper:
        mask  = log_sk < mean_log - (std_log * n_sigma_lower)
    else:
        raise RuntimeError("No flags to process: need to flag upper and/or lower!")

    if pad_mask:
        # also mask either side of a masked value
        mask[1:]  = np.logical_or(mask[1:], mask[0:-1])
        mask[:-1] = np.logical_or(mask[:-1], mask[1:])
    return mask