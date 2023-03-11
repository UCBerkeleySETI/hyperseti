import cupy as cp
import numpy as np
from cusignal import argrelmax
import time

from .utils import on_gpu, datwrapper

# Logging
from .log import get_logger
logger = get_logger('hyperseti.peak')



def find_peaks_argrelmax(data, threshold=20, order=100):
    """ Find peaks using argrelmax on 1D data 
    
    Args:
        data (cp.array): Numpy/cupy array of dedopplerd ata (N_drift, N_beam, N_channel)
        metadata (dict): Metadata dictionary 
        threshold (float): Signal-to-noise threshold. Default 20
        order (int): How many points on each side to use for the comparison in argrelmax
    
    Returns:
        maxvals, maxidx_f, maxidx_dd (list of cp.arrays): SNR values, frequency indexes, dedoppler indexes.

    Notes:
        Uses cusignal.argrelmax to find relative maxima.
        First, the relative maximum is computed along the driftrate axis.
        A mask of data > threshold is also computed; any relative maxima above
        threshold are then returned. 
        We can only find a single maxima per frequency channel using this approach.

        A maxima will not be recorded if it's a plateau, eg. 0001111000, as it's not a relative maxima.
        This means badly-simulated data may not work well.
    """
    t0 = time.time()
    maxvals = data.max(axis=0).squeeze()
    maxmask = maxvals > threshold
    maxidxs = argrelmax(maxvals, order=order)[0]

    # First, find frequency indexes of maxima above threshold 
    # This next line is unusual: 
    # 1) Convert mask from all data into just selected data (above threshold)
    # 2) from maxidxs array we can now use our threshold mask
    maxidx_f = maxidxs[maxmask[maxidxs]]

    # Now we need to find matching dedoppler indexes
    maxidx_dd = cp.argmax(data[:, 0, maxidx_f], axis=0)
    
    # Also find max SNRs
    maxvals = maxvals[maxidx_f]    

    t1 =  time.time()
    logger.info(f"<find_peaks_argrelmax> elapsed time: {(t1-t0)*1e3:2.4f} ms")              
    return maxvals, maxidx_f, maxidx_dd