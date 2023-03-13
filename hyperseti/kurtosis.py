import cupy as cp
import numpy as np

#logging
from .log import get_logger
logger = get_logger('hyperseti.kurtosis')




def spectral_kurtosis(x):
    """ GPU Spectral Kurtosis Kernel 
    
    Args:
        x (array): Data to compute SK on
        N (int): Number of time samples in averaged data
    """
    metadata = x.metadata
    samps_per_sec = (1.0 / np.abs(metadata['frequency_step'])).to('s') / 2 # Nyq sample rate for channel
    N_acc = int(metadata['time_step'].to('s') / samps_per_sec)
    logger.debug(f'rescaling SK by {N_acc}')
    
    x_sum  = cp.sum(x.data, axis=0)
    x2_sum = cp.sum(x.data**2, axis=0)
    n = x.data.shape[0]
    return (N_acc*n+1) / (n-1) * (n*(x2_sum / (x_sum*x_sum)) - 1).squeeze()


def sk_flag(data_array, n_sigma=None, n_sigma_upper=5, n_sigma_lower=5, 
            flag_upper=True, flag_lower=True):
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
    
    Returns:
        mask (np.array, bool): Array of True/False flags per channel
    
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

    var_theoretical = 2.0 / np.sqrt(N_acc)
    std_theoretical = np.sqrt(var_theoretical)
    log_sk   = cp.log2(sk) 
    std_log  = np.abs(np.log2(std_theoretical))
    mean_log = 0
    
    if flag_upper and flag_lower:
        mask  = log_sk < mean_log + (std_log * n_sigma_upper)
        mask  &= log_sk > mean_log - (std_log * n_sigma_lower)
    elif flag_upper and not flag_lower:
        mask  = log_sk > mean_log + (std_log * n_sigma_upper)
    elif flag_lower and not flag_upper:
        mask  = log_sk < mean_log - (std_log * n_sigma_lower)
    else:
        raise RuntimeError("No flags to process: need to flag upper and/or lower!")
    return ~mask