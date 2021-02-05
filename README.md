### hyperseti

A brute-force GPU dedoppler code and hit search package for technosignature searches.

_In beta_

#### Example usage:

```python
import numpy as np
import astropy.units as u
import pylab as plt
from hyperseti import dedoppler
from plotting import imshow_dedopp, imshow_waterfall

# Create a drifting test signal
N_timestep, N_chans = 32, 256
test_data = np.ones(shape=(N_timestep, N_chans ))
for ii in range(N_timestep):
    test_data[ii, N_chans // 2 + ii] = 100

# Create basic metadata
metadata = {'fch1': 1000*u.MHz, 'dt': 1.0*u.s, 'df': 1.0*u.Hz}

# Run dedoppler
dedopp, metadata = dedoppler(test_data, metadata, boxcar_size=1, max_dd=4.0)

# Imshow output
plt.figure(figsize=(8, 3))
plt.subplot(1,2,1)
imshow_waterfall(np.log(test_data), metadata)
plt.subplot(1,2,2)
imshow_dedopp(np.log(dedopp), metadata)
plt.tight_layout()
```

![](https://github.com/UCBerkeleySETI/hyperseti/raw/master/docs/figs/example.png)

Can also search for hits in the dedoppler spectra:

```python
# ... run code from above ...  

from hyperseti import hitsearch
hits = hitsearch(dedopp, metadata, threshold=500)

from plotting import overlay_hits
overlay_hits(hits)
```

| driftrate | f_start | snr | driftrate_idx | channel_idx | boxcar_size |
| --- | --- | --- | --- | --- | --- | 
| 0 	 | 1.0 	 | 1000.000128 	| 3200.0 	| 160 	| 128 	| 1 |

![](https://github.com/UCBerkeleySETI/hyperseti/raw/master/docs/figs/example2.png)

Or the dedoppler and hitsearch can be done in one line with `run_pipeline()`. 
Data can be boxcar averaged to look for wider-band signals, and to retrieve signal-to-noise
for signals with large drift rates:

```python
dedopp, md, hits = run_pipeline(d, metadata, max_dd=1.0, min_dd=None, threshold=100, 
                                    n_boxcar=5, merge_boxcar_trials=True)
```

Reading from file is also supported:

```python
hits = find_et(filename, filename_out='hits.csv', n_parallel=2, gulp_size=2**18, max_dd=1.0, threshold=50)
``

