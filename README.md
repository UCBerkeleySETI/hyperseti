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