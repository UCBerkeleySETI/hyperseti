## hyperseti

<p align="right">
[![Documentation Status](https://readthedocs.org/projects/hyperseti/badge/?version=latest)](https://hyperseti.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/UCBerkeleySETI/hyperseti/branch/master/graph/badge.svg?token=YGW53OTFQA)](https://codecov.io/gh/UCBerkeleySETI/hyperseti)
</p>

A brute-force GPU dedoppler code and hit search package for technosignature searches.


#### Example 1: running the search pipeline on voyager data

```python
from hyperseti.pipeline import find_et

voyager_h5 = '../test/test_data/Voyager1.single_coarse.fine_res.h5'

config = {
    'preprocess': {
        'sk_flag': True,                        # Apply spectral kurtosis flagging
        'normalize': True,                      # Normalize data
        'blank_edges': {'n_chan': 32},          # Blank edges channels
        'blank_extrema': {'threshold': 10000}   # Blank ridiculously bright signals before search
    },
    'dedoppler': {
        'kernel': 'ddsk',                       # Doppler + kurtosis doppler (ddsk)
        'max_dd': 10.0,                         # Maximum dedoppler delay, 10 Hz/s
        'apply_smearing_corr': True,            # Correct  for smearing within dedoppler kernel 
        'plan': 'stepped'                       # Dedoppler trial spacing plan (stepped = less memory)
    },
    'hitsearch': {
        'threshold': 20,                        # SNR threshold above which to consider a hit
    },
    'pipeline': {
        'merge_boxcar_trials': True             # Merge hits at same frequency that are found in multiple boxcars
    }
}

hit_browser = find_et(voyager_h5, config, gulp_size=2**20)
display(hit_browser.hit_table)

hit_browser.view_hit(0, padding=128, plot='dual')

```

#### Example 2: Inspecting a setigen Frame

```python
import numpy as np
import cupy as cp
import pylab as plt
from astropy import units as u

import setigen as stg
from hyperseti.io import from_setigen
from hyperseti.dedoppler import dedoppler
from hyperseti.plotting import imshow_waterfall, imshow_dedopp

# Create data using setigen
frame = stg.Frame(...)

# Convert data into hyperseti DataArray
d = from_setigen(frame)
d.data = cp.asarray(d.data) # Copy to GPU

# Run dedoppler
dedopp_array = dedoppler(d, boxcar_size=1, max_dd=8.0, plan='optimal')

# Plot waterfall / dedoppler
plt.figure(figsize=(8, 3))
plt.subplot(1,2,1)
imshow_waterfall(d)
plt.subplot(1,2,2)
imshow_dedopp(dedopp_array)
plt.tight_layout()
```

![image](https://user-images.githubusercontent.com/713251/164058073-88ccf3b1-b4a1-4160-b650-fca37770f96d.png)

Can also search for hits in the dedoppler spectra:

```python
# ... run code from above ...  

from hyperseti import  hitsearch
hits = hitsearch(dedopp, threshold=100, min_fdistance=10)

from hyperseti.plotting import overlay_hits
imshow_dedopp(dedopp)
overlay_hits(hits)
```

![image](https://user-images.githubusercontent.com/713251/164058025-ab8a3d7a-ffa5-4437-b01b-6c8d6a29cd7c.png)

![image](https://user-images.githubusercontent.com/713251/164058051-9b511f50-d0d0-4058-b512-c062cc7d7964.png)


### Installation

Theoretically, just type:

```
pip install git+https://github.com/ucberkeleyseti/hyperseti
```

hyperseti uses the GPU heavily, so a working CUDA environment is needed, and
requires Python 3.7 or above.
hyperseti relies upon `cupy`, and currently uses a single function, 
`argrelmax` from `cusignal`. These are part of [rapids](https://rapids.ai/start.html)
and are easiest to install using `conda`. If starting from scratch, this should get you most of
the way there:

```
conda create -n hyperseti -c rapidsai -c nvidia -c conda-forge \
    rapids=22.04 python=3.9 cudatoolkit=11.0 dask-sql 

conda activate hyperseti
conda install pandas astropy
pip install logbook setigen blimpy hdf5plugin
pip install git+https://github.com/ucberkeleyseti/hyperseti
```
