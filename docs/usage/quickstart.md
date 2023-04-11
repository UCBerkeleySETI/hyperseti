## Quickstart

### Installation

hyperseti uses the GPU heavily, so a working CUDA environment is needed, and
requires Python 3.7 or above.  hyperseti relies upon `cupy`, which is easiest to install using `conda` (or `mamba`). 

To install from conda/mamba package:

```
conda install -c technosignatures hyperseti
```

If starting from scratch, this should get you most of the way there:

```
conda create -n hyper -c nvidia -c conda-forge python=3.10 cupy jupyterlab ipywidgets
```

Jupyterlab and ipywidgets are optional, but useful for a base environment.

From there:

```
conda activate hyper
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

Hyperseti loads filterbank files in HDF5 and sigproc format, and can also generate a `DataArray` from a [setigen](https://github.com/bbrzycki/setigen) `Frame`. 

```python
from hypersetio.io import load_data         # Loads HDF5/Filterbank/setigen
from hyperseti.io import from_h5            # Loads .h5 HDF5 files
from hyperseti.io import from_fil           # Loads .fil files
from hyperseti.io import from_setigen       # Converts setigen Frame into DataArray

d = from_h5('path/to/data.h5')
```

### Loading 'duck' arrays

A hyperseti `DataArray` can also be created from any array that acts like a Numpy array (i.e. using 'duck typing'), along with a metadata dictionary that defines its axes using `from_metadata()`. Similarly, once you have a `DataArray` you can split out its metadata using
`data_array.split_metadata()`:

```python
metadata_in = {'frequency_start': 1000*u.MHz,
               'time_start': Time(datetime.now()),
               'time_step': 1.0*u.s, 
               'frequency_step': 1.0*u.Hz,
               'dims': ('time', 'beam_id', 'frequency')}

test_data = np.zeros(shape=(16, 1, 2**20), dtype='float32')

d_array = from_metadata(test_data, metadata_in)
```