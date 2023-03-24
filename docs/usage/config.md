## Pipeline config

The main search function, `find_et`, accepts a Python dict as its primary configuration.

The pipeline config dict has for primary keys, each of which holds its own config dict:

* `preprocess`: For pre-processing tasks (normalizing and blanking data).
* `deoppler`: For options controlling dedoppler search.
* `hitsearch`: Searching for hits in dedoppler space.
* `pipeline`: Control and iteration of dedoppler/hitsearch. 


Dicts within the pipeline config dict are passed as kwargs to functions that are called in the pipeline. For example, the `config['dedoppler']` dict is passed with the following invocation:

```python
self.dedopp, self.dedopp_sk = dedoppler(self.data_array, **self.config['dedoppler'])
```

Which passes to the `dedoppler.dedoppler()` method:

```python
def dedoppler(data_array: DataArray, 
              max_dd: u.Quantity, 
              min_dd: u.Quantity=None, 
              boxcar_size: int=1, 
              kernel: str='dedoppler', 
              apply_smearing_corr: bool=False, 
              plan: str='stepped') -> DataArray:
```

A full configuration looks like:

```python
config = {
    'preprocess': {
        'sk_flag': True,                        # Apply spectral kurtosis flagging
        'normalize': True,                      # Normalize data
        'blank_edges': {'n_chan': 32},          # Blank edges channels
        'blank_extrema': {'threshold': 100000}  # Blank ridiculously bright signals before search
    },
    'dedoppler': {
        'kernel': 'ddsk',                       # Doppler + kurtosis doppler (ddsk)
        'max_dd': 5.0,                          # Maximum dedoppler delay, 5 Hz/s
        'min_dd': None,                         # 
        'apply_smearing_corr': False,           # Correct  for smearing within dedoppler kernel 
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
```

### YAML config

Pipeline configuration can also be loaded from a YAML file using the `io.load_config()` method.

```yaml
# example.yaml
dedoppler:
  apply_smearing_corr: false
  kernel: ddsk
  max_dd: 5.0
  min_dd: null
  plan: stepped
hitsearch:
  min_fdistance: null
  threshold: 20
pipeline:
  merge_boxcar_trials: true
  n_boxcar: 10
preprocess:
  blank_edges:
    n_chan: 32
  normalize: true
  sk_flag: true
```