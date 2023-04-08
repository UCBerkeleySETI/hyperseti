## Hitsearch

### Doppler smearing

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
argument in `dedoppler()`. The function that applies this is `filter.apply_boxcar_drift`, which
applies different boxcar filters depending on the drift rate trial:

```python
    # Apply boxcar filter to compensate for smearing
    for boxcar_size in range(2, smearing_nchan_max+1):
        idxs = cp.where(smearing_nchan == boxcar_size)
        # 1. uniform_filter1d computes mean. We want sum, so *= boxcar_size
        # 2. we want noise to stay the same, so divide by sqrt(boxcar_size)
        # combined 1 and 2 give aa sqrt(boxcar_size) factor
        data[idxs] = uniform_filter1d(data[idxs], size=boxcar_size, axis=2) * np.sqrt(boxcar_size)
```

In pipelines, smearing correction can be enabled by setting `config['dedoppler']['apply_smearing_corr']=True`.

Note that S/N calculations will be incorrect if smearing correction is used along with multiple boxcar trials,
so use one approach or the other. Smearing correction will be faster in general than multiple boxcar trials.

### Doppler 'butterflies' 

To find a single 'hit' in dedoppler space requires finding the local maxima, and figuring out which pixels are associated with each local maxima. In dedoppler space, a narrowband signal will produce a 'butterfly' pattern 
(or 'footprint'), which narrows toward a 'waist' at which at the optimal drift rate trial. At this waist point,
signal power is maximally concentrated; as you move away from the waist the signal power is spread out over multiple channels creating the butterfly 'wings'. The butterfly footprint for each hit can cover a large fraction of the search space, particularly when searching for high drift rate signals.

![butterfly](https://user-images.githubusercontent.com/713251/230712277-7aed5bc5-cff3-4684-b220-5f9a7b018f82.png)

#### Signal extent

If the hit is not a perfect narrowband signal, the width of the butterfly waist is determined by the signal's 
intrinsic bandwidth. The `hyperseti.hits.get_signal_extent` method is called on each hit to estimate its extent:
that is, the signals' apparent bandwidth at the optimal dedoppler trial that maximizes signal-to-noise. 

### Iterative blanking

Our local maxima approach to finding hits requires setting a minimum distance within which all other maxima are ignored. This minimum distance means that if there are other bona-fide hits within the excluded zone, they will be missed. Our solution to this is called 'iterative blanking'. After finding the highest S/N hits, we blank these in the original input spectra (i.e. in frequency-time space, where the signals have smaller footprints). The code then re-searches the dedoppler space for other hits.

An example of iterative blanking is show below; in the example, the highest S/N hit is blanked, then the data are searched again. As can be seen, hits hiding below the bright S/N hits become apparent.

![Iterative-blanking](https://user-images.githubusercontent.com/713251/227689177-42e81c48-53cc-4eb9-a8f9-4cea8ce37f2e.png)

Iterative blanking can be enabled by setting `config['pipeline']['n_blank']` to 1 or greater, where `n_blank` is the 
number of iterations to apply. Note that if no new hits are found after blanking, the process will stop. 

### Browsing hits

Results from `find_et` are returned as a `HitBrowser`, which has a `view_hits()` method for viewing this, and an `extract_hits()` method for extracting hits. 

A Pandas DataFrame of all hits found is attached as `hit_brower.hit_table`, and the data are accessible via `hit_browser.data_array`. 

```python
hit_browser = find_et(voyager_h5, config, gulp_size=2**20)
display(hit_browser.hit_table)

hit_browser.view_hit(0, padding=128, plot='dual')
```

![image](https://user-images.githubusercontent.com/713251/227728999-1bec6e2f-bfca-4ab7-ae59-d08010ad8a8d.png)


