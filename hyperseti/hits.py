from copy import deepcopy
import cupy as cp
import numpy as np
import time
import pandas as pd
import os

from astropy import units as u

from .peak import find_peaks_argrelmax
from .kernels.peak_finder import peak_find

from .data_array import DataArray
from .blanking import blank_hit, blank_hits

#logging
from .log import get_logger
logger = get_logger('hyperseti.hits')


def create_empty_hits_table(sk_col=False):
    """ Create empty pandas dataframe for hit data

    Args:
        sk_col (bool): Include a dedoppler spectral kurtosis column (DDSK)
    
    Notes:
        Columns are:
            Driftrate (float64): Drift rate in Hz/s
            f_start (float64): Frequency in MHz at start time
            snr (float64): Signal to noise ratio for detection.
            driftrate_idx (int): Index of array corresponding to driftrate
            channel_idx (int): Index of frequency channel for f_start
            beam_idx (int): Index of beam in which found
            boxcar_size (int): Size of boxcar applied to data
    
    Returns:
        hits (pd.DataFrame): Data frame with columns as above.
    """
    # Create empty dataframe
    cols = {'drift_rate': pd.Series([], dtype='float64'),
                          'f_start': pd.Series([], dtype='float64'),
                          'snr': pd.Series([], dtype='float64'),
                          'driftrate_idx': pd.Series([], dtype='int'),
                          'channel_idx': pd.Series([], dtype='int'),
                          'beam_idx': pd.Series([], dtype='int'),
                          'boxcar_size': pd.Series([], dtype='int'),
                         }
    if sk_col:
        cols['ddsk'] = pd.Series([], dtype='float64')
    hits = pd.DataFrame(cols)
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

    logger.debug(f"merge_hits: sorted hitlist: {p}")
    hits = []
    if len(p) > 1:
        while len(p) > 0:
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
            
            # WAR: If we get tophit with .iloc, it destroys column dtypes as pd.Series does not support mixed dtypes
            # So, here we use a weird indexing that returns a DataFrame of len(1), not a Series
            tophit = pq.sort_values("snr", ascending=False)[0:0+1] 
            # Drop all matched rows
            p = p.drop(pq.index)
            hits.append(tophit)

        hits = pd.concat(hits)
    else:
        hits = p
    t1 = time.time()
    logger.debug(f"merge_hits: Merged {len(hits)}, elapsed time: {(t1-t0)*1e3:2.2f}ms")
    
    return hits

def get_signal_extent(data, d0, p0, g0, threshold=10):
    """ Find the extent (bandwidth) of a signal 
    
    Args:
        data (cp.ndarray): Data array (dedopp, beam_idx, channel)
        g0 (int): Gulp index (channel)
        d0 (int): Dedoppler index
        p0 (int): beam index (pol)
        threshold (float): S/N threshold to use for signal > threshold criterion

    Returns:
        edge_l, edge_u (int, int): Lower and upper offsets before threshold 
    """    
    
    if data[d0, p0, g0] < threshold:
        raise RuntimeError("Signal < threshold. Check indexing is correct")

    N_chan = data.shape[2]

    # Search for upper edge
    eu_found = False
    g0_u = g0
    while not eu_found:
        edge_u = 1
        try:
            while data[d0, p0, g0_u+edge_u] > threshold:
                #print(data_array.data[d0, p0, g0+edge_u])
                edge_u *= 2
                # Have we hit upper limit?
                if g0_u + edge_u > N_chan - 1: 
                    g0_u = N_chan -1
                    eu_found = True
                    break
            if edge_u == 1 or edge_u // 2 == 1:
                eu_found    = True
            g0_u += edge_u // 2
        except IndexError:
            g0_u = N_chan - 1
            eu_found = True


    # Search for lower edge
    el_found = False
    g0_l = g0
    while not el_found:
        edge_l = -1
        try:
            while data[d0, p0, g0_l+edge_l] > threshold:
                edge_l *= 2
                # Have we hit lower limit?
                if g0_l + edge_l <= 0: 
                    g0_l = 0
                    el_found = True
                    break
            if edge_l == -1 or edge_l // 2 == -1:
                el_found = True
            g0_l += edge_l // 2
        except IndexError:
            g0_l = g0
            el_found = True

    return g0_l - g0, g0_u - g0 

def get_signal_extents(dedopp_array, hits, threshold=10):
    """ Find the extent (bandwidth) of hits
    
    Args:
        dedopp_array (DataArray): Data array (dedopp, beam_idx, channel)
        hits (pd.DataFrame)

    Returns:
        edges_l, edges_u (np.array, np.array): Lower and upper offsets before threshold 
    """ 

    edges_l = np.zeros(len(hits), dtype='int32')  
    edges_u = np.zeros(len(hits), dtype='int32')  
    d_col = hits['driftrate_idx']
    p_col = hits['beam_idx'] 
    g_col = hits['gulp_channel_idx']

    for ii in range(len(hits)):
        logger.debug(d_col[ii], p_col[ii], g_col[ii], dedopp_array.data[d_col[ii], p_col[ii], g_col[ii]])
        e_l, e_u = get_signal_extent(dedopp_array.data, d_col[ii], p_col[ii], g_col[ii], threshold=threshold)
        edges_l[ii] = e_l
        edges_u[ii] = e_u 

    return edges_l, edges_u  

def hitsearch(dedopp_array, threshold=10, min_fdistance=100, sk_data=None, **kwargs):
    """ Search for hits using argrelmax method in cusignal
    
    Args:
        dedopp (np.array): Dedoppler search array of shape (N_trial, N_beam, N_chan)
        metadata (dict): Dictionary of metadata needed to convert from indexes to frequencies etc
        threshold (float): Threshold value (absolute) above which a peak can be considered
        min_fdistance (int): Minimum distance in pixels to nearest peak along frequency axis
        sk_data (DataArray): array of 
    
    Returns:
        results (pd.DataFrame): Pandas dataframe of results, with columns 
                                    driftrate: Drift rate in hz/s
                                    f_start: Start frequency channel
                                    snr: signal to noise ratio
                                    driftrate_idx: Index in driftrate array
                                    channel_idx: Index in frequency array
    """
    metadata = deepcopy(dedopp_array.metadata)
    dedopp_data = dedopp_array.data
    
    drift_trials = np.asarray(dedopp_array.drift_rate.data)

    # TODO: Get this out of loop? 
    # Set kernel size to be 2^(N-1)
    #K = 2**(int(np.log2(min_fdistance)))
    #pf = PeakFinder()
    #pf.init(N_chan=dedopp_array.shape[2], N_time=dedopp_array.shape[0], K=K)  


    t0 = time.time()
    dfs = []
    for beam_idx in range(dedopp_data.shape[1]):

        ## TODO: Can we get rid of this copy?
        if dedopp_data.shape[1] > 1:
            imgdata = cp.copy(cp.expand_dims(dedopp_data[:, beam_idx, :].squeeze(), 1))
        else:
            imgdata = dedopp_data.squeeze()

        # Run peak find
        ##intensity, fcoords, dcoords = pf.hitsearch(dedopp_data, beam_id=beam_idx, threshold=threshold, min_spacing=min_fdistance)
        intensity, fcoords, dcoords = peak_find(imgdata, threshold=threshold, min_spacing=min_fdistance)
        #print(intensity, fcoords, dcoords)
        #intensity, fcoords, dcoords = find_peaks_argrelmax(imgdata, threshold=threshold, order=min_fdistance)

        t1 = time.time()
        logger.debug(f"hitsearch: Peak find time: {(t1-t0)*1e3:2.2f}ms")
        t0 = time.time()
        # copy results over to CPU space
        intensity, fcoords, dcoords = cp.asnumpy(intensity), cp.asnumpy(fcoords), cp.asnumpy(dcoords)
        t1 = time.time()
        logger.debug(f"hitsearch: Peak find memcopy: {(t1-t0)*1e3:2.2f}ms")

        t0 = time.time()
        if len(fcoords) > 0:
            driftrate_peaks = drift_trials[dcoords]
            logger.debug(f"hitsearch: {metadata['frequency_start']}, {metadata['frequency_step']}, {fcoords}")
            frequency_peaks = metadata['frequency_start'] + metadata['frequency_step'] * fcoords

            results = {
                'drift_rate': driftrate_peaks,
                'f_start': frequency_peaks,
                'snr': intensity,
                'driftrate_idx': dcoords,
                'channel_idx': fcoords,
                'gulp_channel_idx': np.copy(fcoords), 
                'beam_idx': np.ones_like(fcoords, dtype='int32') * beam_idx
            }

            # Check if we have a slice of data. If so, add slice start point
            # Note: currently assumes slice is only on frequency channel
            if dedopp_array.slice_info is not None:
                ts, bs, fs = metadata['slice_info']
                results['channel_idx'] += fs.start

            # add in spectral kurtosis if computed
            if sk_data is not None:
                if isinstance(sk_data, DataArray):
                    sk_data = sk_data.data
                sk_vals = sk_data[dcoords, beam_idx, fcoords]
                results['ddsk'] = cp.asnumpy(sk_vals)


            # Append numerical metadata keys
            for key, val in metadata.items():
                if isinstance(val, (int, float)):
                    results[key] = val
            
            # Convert into dataframe
            results = pd.DataFrame(results)

            # Compute and append extent (signal bandwidth)
            threshold_ex = np.max((5, threshold/2)) # Minimum 5-sigma, default is 3 dB below detection threshold (i.e. 1/2)
            edges_l, edges_u = get_signal_extents(dedopp_array, results, threshold=threshold_ex) 
            results['extent_lower'] = edges_l
            results['extent_upper'] = edges_u

            dfs.append(results)
        
            t1 = time.time()
            logger.debug(f"hitsearch: Peak find to dataframe: {(t1-t0)*1e3:2.2f}ms")
        try:
            hits = pd.concat(dfs, ignore_index=True)
            #logger.debug(hits)
        except ValueError: # No hits output
            hits = None
        return hits
    else:
        return None
    
