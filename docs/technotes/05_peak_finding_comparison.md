## Peak finding comparisons

This is a gist comparing several approaches to finding peaks.

One of turboseti's limitations is that as maximum drift rate increases, the space 'blanked'
around a hit increases. This means high drift rate searches with turboseti can be blind to
fainter hits hiding behind strong hits.

I tried several approaches from ML clustering and computer vision, but these were too slow.
My current thought is to first find strong hits, then blank these from the data and rerun. 

### Imports

These may or may not be needed and/or out of date
```python
from hyperseti.data_array import from_h5, from_fil
from hyperseti.plotting import imshow_waterfall
from hyperseti import dedoppler, normalize
from hyperseti.peak import find_peaks_argrelmax
from hyperseti.io import from_setigen

import cupy as cp
import numpy as np
import pylab as plt
import time
import pandas as pd
import logging

from astropy import units as u
import setigen as stg
import matplotlib.pyplot as plt

import hdf5plugin
import h5py
from copy import deepcopy
import pyfof
import hyperseti
```

### setigen test data

```python
metadata = {'fch1': 6095.214842353016*u.MHz, 
            'dt': 18.25361108*u.s, 
            'df': 2.7939677238464355*u.Hz}

frame = stg.Frame(fchans=2**12*u.pixel,
                  tchans=32*u.pixel,
                  df=metadata['df'],
                  dt=metadata['dt'],
                  fch1=metadata['fch1'])

test_tones = [
  {'f_start': frame.get_frequency(index=500), 'drift_rate': 0.70*u.Hz/u.s, 'snr': 100, 'width': 20*u.Hz},
  {'f_start': frame.get_frequency(index=700), 'drift_rate': -0.55*u.Hz/u.s, 'snr': 100, 'width': 20*u.Hz},
  {'f_start': frame.get_frequency(index=2048), 'drift_rate': 0.00*u.Hz/u.s, 'snr': 40, 'width': 6*u.Hz},
  {'f_start': frame.get_frequency(index=3000), 'drift_rate': 0.07*u.Hz/u.s, 'snr': 50, 'width': 3*u.Hz}
]

noise = frame.add_noise(x_mean=10, x_std=5, noise_type='chi2')

for tone in test_tones:
    signal = frame.add_signal(stg.constant_path(f_start=tone['f_start'],
                                            drift_rate=tone['drift_rate']),
                          stg.constant_t_profile(level=frame.get_intensity(snr=tone['snr'])),
                          stg.gaussian_f_profile(width=tone['width']),
                          stg.constant_bp_profile(level=1))

fig = plt.figure(figsize=(10, 6))
frame.render()

d = from_setigen(frame)

pd.DataFrame(test_tones)
```


### argrelmax from cusignal

The fastest approach uses argrelmax from cusignal:

```python
@datwrapper(dims=None)
@on_gpu
def find_peaks_argrelmax(data, metadata, threshold=20, order=100):
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
```

```
%timeit find_hits_argrelmax(dd)
2.82 ms ± 20.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

### DBSCAN from cuml

DBSCAN is a clustering algorithm which can be found in `cuml`.

```python
from cuml import DBSCAN

@datwrapper(dims=None)
@on_gpu
def find_hits_dbscan(data, metadata, threshold=20, 
                     eps=10.0, min_samples=1):
    xi, pi, yi = cp.where(data > threshold)
    xiyi = cp.array((xi, yi)).astype('float64').T
    vals_xiyi = data[xi, pi, yi]
    
    clusterer   = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_id = clusterer.fit_predict(xiyi)
    N_clusters  = int(cp.max(cluster_id) + 1)

    centroids_x   = cp.zeros(N_clusters, dtype='int64')
    centroids_y   = cp.zeros(N_clusters, dtype='int64')
    centroids_val = cp.zeros(N_clusters, dtype='float32')

    for cidx in range(N_clusters):
        c     = xiyi[cluster_id == cidx]
        cv    = vals_xiyi[cluster_id == cidx]

        peak_c_idx = cp.argmax(cv)     # Index within cluster
        peak_xiyi_idx = c[peak_c_idx]  # Index within dataframe
        peak_c_val    = cv[peak_c_idx] # Max value

        centroids_x[cidx] = int(peak_xiyi_idx[0])
        centroids_y[cidx] = int(peak_xiyi_idx[1])
        centroids_val[cidx] = peak_c_val
    
    return centroids_val, centroids_y, centroids_x
```

```
%timeit find_hits_dbscan(dd, return_space='cpu')
9.53 ms ± 126 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

### Friends of friends (pyfof)

```python
def find_peaks_fof(data, threshold=10, link_length=10):
    """ Find peaks in image using Friends of Friends algorithm
    
    Uses pyfof.friends_of_friends function
    
    Args:
        data (np.array): 2D Data array (suggest that you normalize first!)
        threshold (float): Data threshold (i.e. signal-to-noise if normalized)
        link_length (float): Linking length between cluster members
    
    """
    
    # Get indexes for 
    xi, yi = np.where(data > threshold)
    vals_xiyi = data[xi, yi]
    
    # PyFoF need (Nx2) array to search through
    # Also needs to be float64 (inefficient recast!)
    xiyi = np.array((xi, yi)).astype('float64').T
    
    # Run Friends of friends algorithm
    groups = pyfof.friends_of_friends(xiyi, link_length)
    
    # For each group, find max value
    centroids_x, centroids_y, centroids_val = [], [], []
    for g in groups:
        # g is a list of indexes of the xiyi array
        # find idx for maximal point in vals_xiyi
        # Note: this is index of and index! (xiyi are indexes of data)
        idx_xiyi      = np.argmax(vals_xiyi[g])
        idx_dmax  = g[idx_xiyi] 
        
        centroids_x.append(xi[idx_dmax])
        centroids_y.append(yi[idx_dmax])
        centroids_val.append(vals_xiyi[idx_dmax])
    return np.array(centroids_val), np.array(centroids_y), np.array(centroids_x)
```

```
%timeit find_peaks_fof(dd.data.squeeze(), threshold=30, link_length=50)
7.06 ms ± 230 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

### Recursive blanking 

```python
def find_max_idx(data):
    """ Find array indexes and maximum value within an array
    
    GPU (cupy) version
    
    Args:
        data (cp.array): Data array
    
    Returngs:
        idx, v: array indexes (tuple) and maximum value v
    """
    idx = cp.unravel_index(cp.argmax(data), data.shape)
    v = data[idx]
    return idx, v

def blank_hit(data, metadata, f0, drate, padding=4):
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
    n_time, n_pol, n_chans = data.shape
    i0     = int((f0 - metadata['frequency_start']) / metadata['frequency_step'])
    i_step =  metadata['time_step'] * drate / metadata['frequency_step']
    i_off  = (i_step * np.arange(n_time) + i0).astype('int64')
    
    min_padding = int(abs(i_step) + 1)  # i_step == frequency smearing
    padding += min_padding 
    i_time = np.arange(n_time, dtype='int64')
    for p_off in range(padding):
        data[i_time, :, i_off] = 0
        data[i_time, :, i_off - p_off] = 0
        data[i_time, :, i_off + p_off] = 0
    return data


@datwrapper()
@on_gpu
def find_hits_recursive(data, metadata, max_hits=100, threshold=20, padding=4):
    """ Find all hits in data, by recursively blanking top hit
    
    This method runs dedoppler multiple times, blanking the brightest hit
    each time then re-running, until max_hits is reached or no more hits
    above threshold are found.
    
    Note:
        max_hits is used to allocate hit arrays, so large values preallocate
        large amounts of memory. (TODO: This could be improved, dynamic resizing)
    
    Args:
        data (cp.array): Data array
        metadata (dict): Metadata with frequency, time info
        max_hits (int): Maximum number of hits
        threshold (float): S/N threshold
        padding (int): Padding to apply when blanking hit -- see blank_hit()
    
    Returns:
        hv, hf, hdr (tuple of cp.array): Values, freq idx, drift rate index arrays. 
    """
    data = normalize(data.astype('float32'), return_space='gpu')

    hit_id = 0
    
    hf = cp.asarray(np.zeros(shape=(max_hits)), dtype='int64')
    hdr = cp.asarray(np.zeros(shape=(max_hits)), dtype='int64')
    hv = cp.asarray(np.zeros(shape=(max_hits)), dtype='float32')
    
    while hit_id < max_hits:
        dd = dedoppler(data, metadata, max_dd=1.0, return_space='gpu')
        idx, val = find_max_idx(dd.data)
        if val > threshold:
            
            f_idx  = int(idx[dd.dims.index('frequency')])
            dr_idx = int(idx[dd.dims.index('drift_rate')])
            hf[hit_id] = f_idx
            hdr[hit_id] = dr_idx
            hv[hit_id] = val
            f0     = dd.frequency[f_idx] * dd.frequency.units
            drate  = dd.drift_rate[dr_idx]  * dd.drift_rate.units
            data = blank_hit(data, metadata, f0, drate, padding=padding)
            hit_id += 1
        else:
            break
    return hv[:hit_id], hf[:hit_id], hdr[:hit_id]
    ```