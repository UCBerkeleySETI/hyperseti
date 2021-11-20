import cupy as cp
import numpy as np
import time
import pandas as pd
import os

from astropy import units as u

from .peak import prominent_peaks
from .data_array import from_fil, from_h5
from .utils import on_gpu, datwrapper

#logging
from .log import logger_group, Logger
logger = Logger('hyperseti.hitsearch')
logger_group.add_logger(logger)


def create_empty_hits_table():
    """ Create empty pandas dataframe for hit data
    
    Notes:
        Columns are:
            Driftrate (float64): Drift rate in Hz/s
            f_start (float64): Frequency in MHz at start time
            snr (float64): Signal to noise ratio for detection.
            driftrate_idx (int): Index of array corresponding to driftrate
            channel_idx (int): Index of frequency channel for f_start
            boxcar_size (int): Size of boxcar applied to data
    
    Returns:
        hits (pd.DataFrame): Data frame with columns as above.
    """
    # Create empty dataframe
    hits = pd.DataFrame({'drift_rate': pd.Series([], dtype='float64'),
                          'f_start': pd.Series([], dtype='float64'),
                          'snr': pd.Series([], dtype='float64'),
                          'driftrate_idx': pd.Series([], dtype='int'),
                          'channel_idx': pd.Series([], dtype='int'),
                          'boxcar_size': pd.Series([], dtype='int'),
                         })
    return hits



def merge_hits(hitlist):
    """ Group hits corresponding to different boxcar widths and return hit with max SNR 
    
    Args:
        hitlist (pd.DataFrame): List of hits
    
    Returns:
        hitlist (pd.DataFrame): Abridged list of hits after merging
    """
    t0 = time.time()
    p = hitlist.sort_values('snr', ascending=False)
    hits = []
    while len(p) > 1:
        # Grab top hit 
        p0 = p.iloc[0]

        # Find channels and driftrates within tolerances
        q = f"""(abs(driftrate_idx - {p0['driftrate_idx']}) <= boxcar_size + 1  |
                abs(driftrate_idx - {p0['driftrate_idx']}) <= {p0['boxcar_size']} + 1)
                & 
                (abs(channel_idx - {p0['channel_idx']}) <= {p0['boxcar_size']} + 1| 
                abs(channel_idx - {p0['channel_idx']}) <= boxcar_size + 1)"""
        q = q.replace('\n', '') # Query must be one line
        pq = p.query(q)
        tophit = pq.sort_values("snr", ascending=False).iloc[0]

        # Drop all matched rows
        p = p.drop(pq.index)
        hits.append(tophit)
    t1 = time.time()
    logger.info(f"Hit merging time: {(t1-t0)*1e3:2.2f}ms")
    
    return pd.DataFrame(hits)


@datwrapper(dims=None)
@on_gpu
def hitsearch(dedopp, metadata, threshold=10, min_fdistance=None, min_ddistance=None):
    """ Search for hits using _prominent_peaks method in cupyimg.skimage
    
    Args:
        dedopp (np.array): Dedoppler search array of shape (N_trial, N_beam, N_chan)
        drift_rates (np.array): List of dedoppler trials corresponding to dedopp N_trial axis
        metadata (dict): Dictionary of metadata needed to convert from indexes to frequencies etc
        threshold (float): Threshold value (absolute) above which a peak can be considered
        min_fdistance (int): Minimum distance in pixels to nearest peak along frequency axis
        min_ddistance (int): Minimum distance in pixels to nearest peak along doppler axis
    
    Returns:
        results (pd.DataFrame): Pandas dataframe of results, with columns 
                                    driftrate: Drift rate in hz/s
                                    f_start: Start frequency channel
                                    snr: signal to noise ratio
                                    driftrate_idx: Index in driftrate array
                                    channel_idx: Index in frequency array
    """
    
    # The following statement used to be:  drift_trials = metadata['drift_rates']
    # and crashed with a KeyError.
    drift_trials = dedopp[:, 0, 0]
    print("DEBUG hitsearch incoming metadata: {}".format(metadata))
    print("DEBUG hitsearch incoming dedopp: {}, 1st 3 in 1st column: {}".format(dedopp.shape, dedopp[0:3, 0, 0].squeeze()))
    print("DEBUG hitsearch drift_trials: {}, 1st 3: {}".format(drift_trials.shape, drift_trials[0:3]))
    
    if min_fdistance is None:
        min_fdistance = metadata['boxcar_size'] * 2
    
    if min_ddistance is None:
        min_ddistance = len(drift_trials) // 4

    print("DEBUG min_fdistance = {}, min_ddistance = {}".format(min_fdistance, min_ddistance))
    
    t0 = time.time()
    dfs = []
    for beam_idx in range(dedopp.shape[1]):
        imgdata = dedopp[:, beam_idx, :].squeeze()
        intensity, fcoords, dcoords = prominent_peaks(imgdata, min_xdistance=min_fdistance, min_ydistance=min_ddistance, threshold=threshold)
        print("DEBUG intensity = {}, fcoords = {}, dcoords = {}".format(intensity, fcoords, dcoords))
        t1 = time.time()
        logger.info(f"Peak find time: {(t1-t0)*1e3:2.2f}ms")
        t0 = time.time()
        # copy results over to CPU space
        intensity, fcoords, dcoords = cp.asnumpy(intensity), cp.asnumpy(fcoords), cp.asnumpy(dcoords)
        t1 = time.time()
        logger.info(f"Peak find memcopy: {(t1-t0)*1e3:2.2f}ms")

        t0 = time.time()
        if len(fcoords) > 0:
            driftrate_peaks = drift_trials[dcoords]
            print("DEBUG hitsearch: driftrate_peaks: {}".format(driftrate_peaks))
            logger.debug(f"{metadata['frequency_start']}, {metadata['frequency_step']}, {fcoords}")
            frequency_peaks = metadata['frequency_start'] + metadata['frequency_step'] * fcoords
            print("DEBUG hitsearch: frequency_peaks: {}".format(frequency_peaks))

            results = {
                'drift_rate': driftrate_peaks,
                'f_start': frequency_peaks,
                'snr': intensity,
                'driftrate_idx': dcoords,
                'channel_idx': fcoords,
                'beam_idx': beam_idx
            }

            # Append numerical metadata keys
            for key, val in metadata.items():
                if isinstance(val, (int, float)):
                    results[key] = val

            print("hitsearch: results type {}, value: {}".format(type(results), results))
            df = pd.DataFrame.from_dict(results)
            dfs.append(df)
        
            t1 = time.time()
            logger.info(f"Peak find to dataframe: {(t1-t0)*1e3:2.2f}ms")
        try:
            hits = pd.concat(dfs)
            logger.debug(hits)
        except ValueError: # No hits output
            hits = None
        return hits
    else:
        return None
    
