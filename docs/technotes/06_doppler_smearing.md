## Doppler smearing

If a narrowband signal drifts too fast, its power will be smeared across frequency bins.

For each time step, the amount of smearing expected (in number of channels) is given by:
```
N = integration_time * drift_rate / channel_bw
```
For example, a signal with a 10 Hz/s drift rate will drift across 10x channels of 1 Hz width
in 1 second integration. 

This leads to a O(N) loss in sensitivity, unless smearing is accounted for. In incoherent 
searches (i.e. on power spectra), a sqrt(N) factor can be retrieved by averaging neighboring
channels together. 

In `hyperseti`, a smearing correction can be applied using the `apply_smearing_correction=True` 
argument in `dedoppler()`. The function that does this is `filter.apply_boxcar_drift`. This
function applies different boxcar filters depending on the drift rate trial:

```python
    # Apply boxcar filter to compensate for smearing
    for boxcar_size in range(2, smearing_nchan_max+1):
        idxs = cp.where(smearing_nchan == boxcar_size)
        # 1. uniform_filter1d computes mean. We want sum, so *= boxcar_size
        # 2. we want noise to stay the same, so divide by sqrt(boxcar_size)
        # combined 1 and 2 give aa sqrt(boxcar_size) factor
        data[idxs] = uniform_filter1d(data[idxs], size=boxcar_size, axis=2) * np.sqrt(boxcar_size)
```