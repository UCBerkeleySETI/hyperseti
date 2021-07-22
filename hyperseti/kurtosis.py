import cupy as cp
import numpy as np

from .utils import on_gpu, datwrapper

#logging
from .log import logger_group, Logger
logger = Logger('hyperseti.kurtosis')
logger_group.add_logger(logger)



@on_gpu
@datwrapper(dims=None)
def spectral_kurtosis(x, metadata):
    """ GPU Spectral Kurtosis Kernel 
    
    Args:
        x (array): Data to compute SK on
        N (int): Number of time samples in averaged data
    """
    samps_per_sec = (1.0 / np.abs(metadata['frequency_step'])).to('s') / 2 # Nyq sample rate for channel
    N_acc = int(metadata['time_step'].to('s') / samps_per_sec)
    logger.debug(f'rescaling SK by {N_acc}')
    
    x_sum  = cp.sum(x, axis=0)
    x2_sum = cp.sum(x**2, axis=0)
    n = x.shape[0]
    return (N_acc*n+1) / (n-1) * (n*(x2_sum / (x_sum*x_sum)) - 1).squeeze()


@on_gpu
@datwrapper(dims=None)
def sk_flag(data, metadata, n_sigma_upper=3, n_sigma_lower=2, 
            flag_upper=True, flag_lower=True):
    """ Apply spectral kurtosis flagging 
    
    Args:
        data (np.array): Numpy array with shape (N_timestep, N_beam, N_channel)
        metadata (dict): Metadata dictionary, should contain 'df' and 'dt'
                         (frequency and time resolution)
        boxcar_mode (str): Boxcar mode to apply. mean/sum/gaussian.
        n_sigma_upper (float): Number of stdev above SK estimate to flag (upper bound)
        n_sigma_lower (float): Number of stdev below SK estmate to flag (lower bound)
        flag_upper (bool): Flag channels with large SK (highly variable signals)
        flag_lower (bool): Flag channels with small SK (very stable signals)
        return_space ('cpu' or 'gpu'): Returns array in CPU or GPU space
    
    Returns:
        mask (np.array, bool): Array of True/False flags per channel
    """
    Fs = (1.0 / metadata['frequency_step'] / 2)
    samps_per_sec = np.abs(Fs.to('s').value) # Nyq sample rate for channel
    N_acc = int(metadata['time_step'].to('s').value / samps_per_sec)

    var_theoretical = 2.0 / np.sqrt(N_acc)
    std_theoretical = np.sqrt(var_theoretical)
    sk = spectral_kurtosis(data, metadata)
    
    if flag_upper and flag_lower:
        mask  = sk > 1.0 + n_sigma_upper * std_theoretical
        mask  |= sk < 1.0 - (n_sigma_lower * std_theoretical)
    elif flag_upper and not flag_lower:
        mask  = sk > 1.0 + n_sigma_upper * std_theoretical
    elif flag_lower and not flag_upper:
        mask  = sk < 1.0 - (n_sigma_lower * std_theoretical)
    else:
        raise RuntimeError("No flags to process: need to flag upper and/or lower!")
    return mask