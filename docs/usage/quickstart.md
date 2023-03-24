## Quickstart

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

### Example pipeline

```python
from hyperseti.pipeline import find_et

voyager_h5 = '../test/test_data/Voyager1.single_coarse.fine_res.h5'

config = {
    'preprocess': {
        'sk_flag': True,                        # Apply spectral kurtosis flagging
        'normalize': True,                      # Normalize data
        'blank_edges': {'n_chan': 32},          # Blank edges channels
        'blank_extrema': {'threshold': 10000}   # Blank ridiculously bright signals before search
        'poly_fit': 5                     
    },
    'dedoppler': {
        'kernel': 'ddsk',                       # Doppler + kurtosis doppler (ddsk)
        'max_dd': 10.0,                          # Maximum dedoppler delay, 5 Hz/s
        'min_dd': None,                         # 
        'apply_smearing_corr': True ,           # Correct  for smearing within dedoppler kernel 
                                                # Note: set to False if using multiple boxcars 
        'plan': 'stepped'                       # Dedoppler trial spacing plan (stepped = less memory)
    },
    'hitsearch': {
        'threshold': 20,                        # SNR threshold above which to consider a hit
        'min_fdistance': None                   # Automatic calculation of min. channel spacing between hits
    },
    'pipeline': {
        'n_boxcar': 10,                         # Number of boxcar trials to apply (10 stages, 2^10 channels)
                                                # Boxcar is a moving average to compensate for smearing / broadband
        'merge_boxcar_trials': True             # Merge hits at same frequency that are found in multiple boxcars
    }
}

hit_browser = find_et(voyager_h5, config, 
                gulp_size=2**18,  # Note: intentionally smaller than 2**20 to test slice offset
                filename_out='./test_voyager_hits.csv',
                log_output=True,
                log_config=True
                )

# find_et returns a hit browser object that makes it easy to plot hits 
print(hit_browser.hit_table)

hit_browser.view_hit(0, padding=128, plot='dual')
```

### Loading files

```python
from hyperseti.io import from_h5

d = from_h5('path/to/data.h5')
```
