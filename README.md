### hyperseti

A brute-force GPU dedoppler code and hit search package for technosignature searches.

_In beta_

#### Example usage:

```python
import numpy as np
import pylab as plt
from astropy import units as u

import setigen as stg
from hyperseti.io import from_setigen
from hyperseti import dedoppler
from hyperseti.plotting import imshow_waterfall, imshow_dedopp

# Create data using setigen
frame = stg.Frame(fchans=8192*u.pixel,
                  tchans=32*u.pixel,
                  df=2*u.Hz,
                  dt=10*u.s,
                  fch1=1420*u.MHz)
noise = frame.add_noise(x_mean=10, noise_type='chi2')
signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(index=2000),
                                            drift_rate=2*u.Hz/u.s),
                          stg.constant_t_profile(level=frame.get_intensity(snr=50)),
                          stg.gaussian_f_profile(width=100*u.Hz),
                          stg.constant_bp_profile(level=1))

# Convert data into hyperseti DataArray
d = from_setigen(frame)

# Run dedoppler
dedopp, md = dedoppler(d, boxcar_size=1, max_dd=8.0)

# Plot waterfall / dedoppler
plt.figure(figsize=(8, 3))
plt.subplot(1,2,1)
imshow_waterfall(d)
plt.subplot(1,2,2)
imshow_dedopp(dedopp)
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
```

