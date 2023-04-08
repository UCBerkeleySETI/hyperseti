## Storing hits with HitDatabase 

Hyperseti implements a simple HDF5-backed database for storing hits from multiple observations. These files can be accessed via the `hyperseti.io.hit_db.HitDatabase` class. 

The `HitDatabase` class provides:

* `hit_db.list_obs()` - List all observations within the database file.
* `hit_db.add_obs()`  - Add a new observation to database.
* `hit_db.get_obs()`  - Retrieve hit data table from the database.
* `hit_db.get_obs_schema()` - Retrieve a description of the columns based on hit_db schema
* `hit_db.get_obs_config()` - Retrieve the observation pipeline config YAML (if present)
* `get_obs_metadata()` - Retrieve metadata about an observation.
* `hit_db.browse_obs()` - Return a `HitBrowser` object for given obs_id.

The last of these, `browse_obs()` is very useful and probably the main method of note for most data analysis.

#### Data model notes:

Within the HDF5 file, a new group is created in the root for each observation.
Table data are a column store, one dataset per column, i.e.:

```
GROUP "/" {
    GROUP "proxima_cen" {
        DATASET "beam_idx"
        DATASET "boxcar_size"
        DATASET "snr"
        DATASET ...}
    GROUP "alpha_cen" {
        DATASET "beam_idx"
        DATASET "boxcar_size"
        DATASET "snr"
        DATASET ...}
    GROUP ... {
        DATASET ...
    }  
}
```

The file format can be identified by the following attributes in the root group:

```
CLASS = 'HYPERSETI_DB'
VERSION = 'X.Y.Z'
```

The database may optionally provide paths to data files for each observation, and a copy of the
YAML [pipeline config](https://hyperseti.readthedocs.io/en/latest/usage/config.html) used when running the pipeline. Any additional metadata are stored as attributes. 


#### Database schema

A description of the hit database columns is provided in `hyperseti.io.hit_db_schema.yml`, reproduced below:

```yaml
# Version 1.0.0
# Hit details - primary
snr:
  description: Signal-to-noise ratio of detected hit
  dtype: float32
  mandatory: true
f_start:
  description: Frequency of detected hit at first timestep (MHz)
  dtype: float64
  mandatory: true
  units: MHz
drift_rate:
  description: Drift rate trial for detected hit (Hz/s)
  dtype: float64
  units: Hz/s
  mandatory: true

# Primary data indexes
channel_idx:  
  description: Index of hit on channel axis
  dtype: int32
  mandatory: true
gulp_channel_idx:  
  description: Index of hit within gulp, on channel axis
  dtype: int32
  mandatory: true
beam_idx:  
  description: Index of hit within gulp, on beam/pol axis
  dtype: int32
  mandatory: true
driftrate_idx:
  description: Index of hit within gulp, on drift rate axis
  dtype: int32
  mandatory: true

# Hit details - secondary
extent_lower:
  description: Extent of detected signal in doppler space, lower extent (# channels)
  dtype: int32
  mandatory: true
extent_upper:  
  description: Extent of detected signal in doppler space, upper extent (# channels)
  dtype: int32
  mandatory: true
ddsk:
  description: Spectral Kurtosis computed along dedoppler path (DDSK)
  dtype: float32
  mandatory: false

# Gulp information
gulp_idx:
  description: Index of gulp from within the data array
  dtype: int64
  mandatory: true
gulp_size:
  description: Size of each gulp
  dtype: int64
  mandatory: true
hits_in_gulp:
  description: Number of hits detected in gulp
  dtype: int32
  mandatory: false

# Pipeline information
n_integration:
  description: Number of timestep within the file
  dtype: int32
  mandatory: true
n_blank:
  description: Number of signal blanking iterations in search pipeline
  dtype: int32
  mandatory: false
blank_count:
  description: Count (index) for signal blanking iteration
  dtype: int32
  mandatory: false
boxcar_size:
  description: Size of boxcar mean filter used before hit search
  dtype: int32
  mandatory: true

# Preprocessing information 
# Placehold vars: X runs from [0, N_beam], Y runs from [0, N_poly_coeffs]
n_poly:
  description: Number of coefficients in polynomial bandpass fit
  dtype: int32
  mandatory: false
bX_gulp_mean:
  description: Mean computed during preprocessing normalization for beam X (bX)
  dtype: float32
  mandatory: false
bX_gulp_std:
  description: Stdev computed during preprocessing normalization for beam X (bX)
  dtype: float32
  mandatory: false
bX_gulp_flag_frac:
  description: Fraction of data flagged with SK flagger during normalization for beam X (bX)
  dtype: float32
  mandatory: false
bX_gulp_poly_cY:
  description: Polynomial fit coefficient Y (cY), beam X (bX)
  dtype: float32
  mandatory: false
```


