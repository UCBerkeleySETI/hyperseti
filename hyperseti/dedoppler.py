import cupy as cp
import numpy as np
import time
import os
import sys
from copy import deepcopy

from astropy import units as u
from cupyx.scipy.ndimage import uniform_filter1d

from .kernels.dedoppler import dedoppler_kernel, dedoppler_kurtosis_kernel, dedoppler_with_kurtosis_kernel
from .filter import apply_boxcar
from .data_array import from_metadata, DataArray
from .dimension_scale import DimensionScale, ArrayBasedDimensionScale

#logging
from .log import get_logger
logger = get_logger('hyperseti.dedoppler')

def calc_delta_dd(data_array: DataArray) -> (float, float):
    """ Compute minimum dedoppler drift delta (Step size) for a given array
    
    Args:
        data_array (DataArray): Data array with (time, beam_id, frequency) axes
    
    Returns:
        obs_len  (float): Observation length in seconds
        delta_dd (float): Minimum step size in dedoppler space
     """
    obs_len  = data_array.time.elapsed.to('s').value
    delta_dd = data_array.frequency.step.to('Hz').value / obs_len  # e.g. 2.79 Hz / 300 s = 0.0093 Hz/s
    return obs_len, delta_dd


def calc_ndrift(data_array: DataArray, max_dd: u.Quantity) -> int:
    """ Calculate the number of channels a drifting signal will cross 

    Args:
        data_array (DataArray): Data array with 'frequency' and 'time' dimensions)
        max_dd (astropy.units.Quantity): Maximum drift rate to consider (in Hz/s)

    Returns:
        min_fdistance (int): Number of frequency channels spanned by a signal with
                             drift rate equal to max_dd
    """
    deltaf = data_array.frequency.step.to('Hz').value
    deltat = data_array.time.step.to('s').value
    max_dd = max_dd.to('Hz/s').value if isinstance(max_dd, u.Quantity) else max_dd
    n_int  = data_array.shape[0]

    min_fdistance = int(np.abs(deltat * n_int * max_dd / deltaf))
    return min_fdistance


def apply_boxcar_drift(data_array: DataArray):
    """ Apply boxcar filter to compensate for doppler smearing
    
    An optimal boxcar is applied per row of drift rate. This retrieves
    a sensitivity increase of sqrt(boxcar_size) for a smeared signal.
    (Still down a sqrt(boxcar_size) compared to no smearing case).
    
    Args:
        data_array (DataArray): Array to apply boxcar filters to
    
    Returns:
         data_array (DataArray): Array with boxcar filters applied.
    """
    logger.debug(f"apply_boxcar_drift: Applying moving average based on drift rate.")
    metadata = data_array.metadata
    
    # Note: dedoppler array no longer has time dimensions, so need to read
    # metadata stores in attributes (integration_time and n_integration)
    drates = cp.asarray(data_array.drift_rate.data)
    df = data_array.frequency.step.to('Hz').value
    dt = metadata['integration_time'].to('s').value / metadata['n_integration']

    # Compute smearing (array of n_channels smeared for given driftrate)
    smearing_nchan = cp.abs(dt * drates / df).astype('int32')
    smearing_nchan_max = cp.asnumpy(cp.max(smearing_nchan))

    # Apply boxcar filter to compensate for smearing
    boxcars = map(int, list(cp.asnumpy(cp.unique(smearing_nchan))))
    for boxcar_size in boxcars:
        idxs = cp.where(smearing_nchan == boxcar_size)
        # 1. uniform_filter1d computes mean. We want sum, so *= boxcar_size
        # 2. we want noise to stay the same, so divide by sqrt(boxcar_size)
        # combined 1 and 2 give aa sqrt(2) factor
        logger.debug(f"boxcar_size: {boxcar_size}, dedopp idxs: {idxs}")
        if boxcar_size > 1:
            data_array.data[idxs] = uniform_filter1d(data_array.data[idxs], size=boxcar_size, axis=2) * np.sqrt(boxcar_size)
    return data_array


def plan_stepped(N_time: int, N_dopp_lower: int, N_dopp_upper: int) -> np.array:
    """ Create dedoppler plan where step size doubles after N_time
    
    This plan is good for large drift rate search ranges, as it requires
    log2(N) fewer trials.  
    
    Args:
        N_time (int): Number of time integrations in data array
        N_dopp_lower (int): Minimum drift rate correction in # channels
        N_dopp_upper (int): Maximum drift rate correction in # channels
    
    Returns
        steps (np.array): Numpy array of dedoppler trials as integer steps
    """
    reverse = False
    if N_dopp_lower > N_dopp_upper:
        reverse = True
        N_dopp_lower, N_dopp_upper = N_dopp_upper, N_dopp_lower
    N_stages, step, curval = 0, 1, 0
    steps_upper = []
    while curval < N_dopp_upper:
        if curval >= N_dopp_lower:
            steps_upper.append(np.arange(N_time, dtype='int32') * step + curval)
        curval += N_time * step
        #print(curval, step, N_stages)
        step *=2
        N_stages += 1
        
    steps_lower = []
    N_stages, step, curval = 0, 1, 0
    while curval > N_dopp_lower:
        if curval <= N_dopp_upper:
            steps_lower.append(-np.arange(N_time, dtype='int32') * step + curval)
        curval -= N_time * step
        #print(curval, step, N_stages)
        step *= 2
        N_stages += 1
    
    if len(steps_upper) > 0:
        steps = steps_upper
        if len(steps_lower) > 0:
            steps += steps_lower 
    elif len(steps_lower) > 0:
        steps = steps_lower
    else:
        raise RuntimeError("No steps!")
    steps = np.unique(np.concatenate(steps))
    steps.sort()
    if reverse:
        steps = steps[::-1]
    return steps


def plan_optimal(N_time: int, N_dopp_lower: int, N_dopp_upper: int) -> np.array:
    """ Basic dedoppler plan to check every possible drift correction 
    
    Args:
        N_time (int): Number of time integrations in data array
        N_dopp_lower (int): Minimum drift rate correction in # channels
        N_dopp_upper (int): Maximum drift rate correction in # channels
    
    Returns:
        dd_shifts (np.array): Array of dedoppler trials as integer steps
    """
    if N_dopp_upper > N_dopp_lower:
        dd_shifts      = np.arange(N_dopp_lower, N_dopp_upper + 1, dtype='int32')
    else:
        dd_shifts      = np.arange(N_dopp_upper, N_dopp_lower + 1, dtype='int32') [::-1]
    return dd_shifts


def dedoppler(data_array: DataArray, max_dd: u.Quantity, min_dd: u.Quantity=None, boxcar_size: int=1, 
              kernel: str='dedoppler', apply_smearing_corr: bool=False, plan: str='stepped') -> DataArray:
    """ Apply brute-force dedoppler kernel to data
    
    Args:
        data_array (DataArray): DataArray with shape (N_timestep, N_beam, N_channel)
        max_dd (astropy.Quantity): Maximum doppler drift in Hz/s to search out to.
        min_dd (astropy.Quantity): Minimum doppler drift to search. 
                                   If set to None (default), it will use -max_dd (not 0)!
        kernel (str): 'dedoppler' or 'kurtosis' or 'ddsk'
        plan (str): Dedoppler plan to use. One of 'full' or 'stepped'
    
    Dedoppler plans:
        'optimal':    Does every possible trial between min_dd and max_dd
        'stepped': Doubles trial after every N_time, based on data_array shape 
                   E.g. for N_time = 2 trials are [0, 1, 2, 4, 8, 16, ..]
    
    Returns:
        dedopp_array (DataArray): Array of dedoppler space data
        dedopp_sk_array (DataArray): If kernel='ddsk', a dedopp_sk DataArray is also returned
    """
   
    t0 = time.time()

    metadata = deepcopy(data_array.metadata)

    if min_dd is None:
        min_dd = np.abs(max_dd) * -1
    else:
        logger.debug(f"dedoppler: Minimum dedoppler rate supplied: {min_dd} Hz/s")
    
    # Compute minimum possible drift (delta_dd)
    N_time, N_beam, N_chan = data_array.data.shape
    obs_len, delta_dd = calc_delta_dd(data_array)
    
    # Compute dedoppler shift schedules
    N_dopp_upper   = int(max_dd / delta_dd)
    N_dopp_lower   = int(min_dd / delta_dd)

    if max_dd == 0 and min_dd is None:
        dd_shifts = np.array([0], dtype='int32')
    else:
        plans = {'optimal': plan_optimal,
                 'stepped': plan_stepped}

        plan_func = plans.get(plan)
        dd_shifts = plan_func(N_time, N_dopp_lower, N_dopp_upper)
    
    # Correct for negative frequency step
    if data_array.frequency.val_step < 0:
        dd_shifts *= -1

    logger.debug("dedoppler: delta_dd={}, N_dopp_upper={}, N_dopp_lower={}, dd_shifts={}"
                 .format(delta_dd, N_dopp_upper, N_dopp_lower, dd_shifts))

    dd_shifts_gpu  = cp.asarray(dd_shifts)
    N_dopp = len(dd_shifts)
    
    # Allocate GPU memory for dedoppler data
    dedopp_gpu = cp.zeros((N_dopp, N_beam, N_chan), dtype=cp.float32)
    if kernel == 'ddsk':
        dedopp_sk_gpu = cp.zeros((N_dopp, N_beam, N_chan), dtype=cp.float32)
    t1 = time.time()
    logger.debug(f"dedoppler: setup time: {(t1-t0)*1e3:2.2f}ms")

    # TODO: Candidate for parallelization
    for beam_id in range(N_beam):

        # Select out beam
        d_gpu = data_array.data[:, beam_id, :] 

        # Launch kernel
        t0 = time.time()

        # Apply boxcar filter
        if boxcar_size > 1:
            d_gpu = apply_boxcar(d_gpu, boxcar_size=boxcar_size, mode='gaussian')

        # Allocate GPU memory for dedoppler data
        if N_beam > 1:
            _dedopp_gpu = cp.zeros((N_dopp, N_chan), dtype=cp.float32)
            if kernel == 'ddsk':
                _dedopp_sk_gpu = cp.zeros((N_dopp, N_chan), dtype=cp.float32)
        else:
            _dedopp_gpu = dedopp_gpu.squeeze()
            if kernel == 'ddsk':
                _dedopp_sk_gpu = dedopp_sk_gpu.squeeze()
        
        # Setup grid and block dimensions
        F_block = np.min((N_chan, 1024))
        F_grid  = N_chan // F_block
        #print(dd_shifts)
        logger.debug(f"dedoppler: Kernel shape (grid, block) {(F_grid, N_dopp), (F_block,)}")

        # Calculate number of integrations within each (time-averaged) channel
        samps_per_sec = np.abs((1.0 / data_array.frequency.step).to('s').value) / 2 # Nyq sample rate for channel
        N_acc = int(data_array.time.step.to('s').value / samps_per_sec)
        
        if kernel == 'dedoppler':
            logger.debug(f"{type(d_gpu)}, {type(_dedopp_gpu)}, {N_chan}, {N_time}")
            dedoppler_kernel((F_grid, N_dopp), (F_block,), 
                            (d_gpu, _dedopp_gpu, dd_shifts_gpu, N_chan, N_time)) # grid, block and arguments
        elif kernel == 'kurtosis':
            # output must be scaled by N_acc, which can be figured out from df and dt metadata
            logger.debug(f'dedoppler kurtosis: rescaling SK by {N_acc}')
            logger.debug(f"dedoppler kurtosis: driftrates: {dd_shifts}")
            dedoppler_kurtosis_kernel((F_grid, N_dopp), (F_block,), 
                            (d_gpu, _dedopp_gpu, dd_shifts_gpu, N_chan, N_time, N_acc)) # grid, block and arguments 
        elif kernel == 'ddsk':
            # output must be scaled by N_acc, which can be figured out from df and dt metadata
            logger.debug(f'dedoppler ddsk: rescaling SK by {N_acc}')
            logger.debug(f"dedoppler ddsk: driftrates: {dd_shifts}")
            dedoppler_with_kurtosis_kernel((F_grid, N_dopp), (F_block,), 
                            (d_gpu, _dedopp_gpu, _dedopp_sk_gpu, dd_shifts_gpu, N_chan, N_time, N_acc)) 
                            # grid, block and arguments
        else:
            logger.critical("dedoppler: Unknown kernel={} !!".format(kernel))
            sys.exit(86)
    
        t1 = time.time()
        logger.debug("dedoppler: kernel ({}) time {:2.2f}ms".format(kernel, (t1-t0)*1e3))

        if N_beam == 1:
            dedopp_gpu = cp.expand_dims(_dedopp_gpu, axis=1)
        else:
            dedopp_gpu[:, beam_id] = _dedopp_gpu
    
    # Create output DataArray
    output_dims = ('drift_rate', 'beam_id', 'frequency')
    output_units = data_array.units

    output_scales = {
        'drift_rate': ArrayBasedDimensionScale('drift_rate', dd_shifts * delta_dd, 'Hz/s'),
        'beam_id': data_array.beam_id,
        'frequency': data_array.frequency
        }

    output_attrs = deepcopy(data_array.attrs)
    output_attrs['boxcar_size'] = boxcar_size
    output_attrs['n_integration'] = N_time
    output_attrs['integration_time'] = data_array.time.elapsed

    # need to pass on slice_info and parent_shape if they are set on the input array
    slice_info   = data_array.slice_info
    parent_shape = data_array.parent_shape
    dedopp_array = DataArray(dedopp_gpu, output_dims, output_scales, output_attrs,
                             units=output_units, slice_info=slice_info, parent_shape=parent_shape)

    #logger.debug("metadata={}".format(metadata))

    if apply_smearing_corr:
        # Note: do not apply smearing corr to DDSK
        logger.debug(f"dedoppler: Applying smearing correction")
        dedopp_array = apply_boxcar_drift(dedopp_array)

    if kernel == 'ddsk':
        dedopp_sk_array = DataArray(dedopp_sk_gpu, output_dims, output_scales, output_attrs, 
                                    units=output_units, slice_info=slice_info, parent_shape=parent_shape)
        return dedopp_array, dedopp_sk_array
    else:
        return dedopp_array


def dedoppler_simple(data_array: DataArray, drift_rate: u.Quantity):
    """ Simple dedoppler algorithm for removing a given drift rate
    
    Args:
        data_array (DataArray): Data with (time, beam, frequency) axes
        drift_rate (Quantity | float): Drift rate to compensate for
    
    Returns:
        data_array (DataArray): Dedoppler'd data array
    """


    n_drift = calc_ndrift(data_array, drift_rate)
    n_t     = data_array.data.shape[0]
    
    for bb in range(data_array.data.shape[1]):
        for ii in range(1, n_t):
            n_roll = int((ii / n_t) * n_drift)
            data_array.data[ii, bb] = np.roll(data_array.data[ii, bb], n_roll - n_drift // 2)

    return data_array