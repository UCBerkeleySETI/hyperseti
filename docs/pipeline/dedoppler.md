## Dedoppler

Hyperseti uses a brute-force dedoppler GPU kernel, which can also compute the 'DDSK' dedoppler spectral kurtosis (see below).

### DDSK Kernel

Following on from [technote #4](https://github.com/UCBerkeleySETI/hyperseti/blob/master/docs/technotes/04_sk_flag.md),
drifting tones cross frequency bins, so each bin will register a high kurtosis value. If a
tone had zero drift, in contrast, its constituent frequency bin would have a near-zero kurtosis value.

We can compute the SK estimator along different drift rates. Here's example with a time-varying signal and constant tone, with spectral kurtosis computed along different drift rate trials.

![image](https://user-images.githubusercontent.com/713251/115005455-751fa780-9eda-11eb-9456-c536ec47e54c.png)

Unlike regular dedoppler summation, the 'peak' is zero for the constant tone (low SK), and large for the varying SK. 
This gives us extra information 

The ['ddsk' kernel](https://github.com/UCBerkeleySETI/hyperseti/blob/master/hyperseti/kernels/dedoppler.py), short for for 'de-dopplered spectral kurtosis' computes SK and regular dedoppler sum at the same time. A 'ddsk' column is added. For example, from a Voyager 1 run:

```
     drift_rate      f_start           snr  driftrate_idx  channel_idx  beam_idx  boxcar_size      ddsk  n_integration
6    -0.000000  8419.921871  48011.296875            0.0     524289.0       0.0          4.0  0.000373           16.0
8    -0.382660  8419.297025    447.540619           40.0     747930.0       0.0          4.0  0.385157           16.0
9    -0.382660  8419.274369     61.139175           40.0     756039.0       0.0          4.0  5.339909           16.0
7    -0.373093  8419.319360     59.822662           39.0     739936.0       0.0          4.0  4.278475           16.0
```

Here we have found:
* the DC bin (6) with a very low ddk at zero drift.
* The carrier (8), with a ddsk < 1 (less than chi-squared expectation), meaning it's somewhat stable
* The two sidebands (9 and 7), which have high ddsk values, due to the modulation.

A bit of real-world testing will be important here to verify when it's useful. But the following should hold:
* 'drifting squiggles' due to unstable LOs will have high ddsk.
* local bright impulsive RFI will have high ddsk.
* stable drifting tones will have ddsk < 1
* really low values of ddsk probably mean it's local RFI (we would expect some scintillation?).


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

### Drift rate plans

The brute-force dedoppler kernel used in hyperseti allows for any set of drift rates to be searched -- they do not have to be uniformly spaced. There are currently two drift rate plans that can be used: `'optimal'` or `'stepped'`. We recommend `'stepped'` for searches and `'optimal'` for follow-up.

#### Optimal plan

In the optimal plan, the drift rate step is set to the dedoppler resolution. The smallest step possible in the dedoppler search is for a signal that drifts across one channel over the full observation; that is:

```
dd = df / t_obs
```

where `dd` is the dedoppler resolution in Hz/s, `df` is the frequency resolution in Hz, and `t_obs` is the total observation time. For example, for a 5-minute (300 s) observation with 3 Hz channel resolution, the dedoppler resolution is 0.01 Hz/s. 

The optimal plan searches every possible dedoppler trial up to `max_dedopp` (in Hz/s), so the the number of trials required (`N_dd`) is 


```
N_dd = max_dedopp / dd
```

For example, a search up to 10 Hz/s with 0.01 Hz/s resolution requires 1000 trials. The memory required to store an array with `N_dd` trials and `N_chan` channels in float32 data is `N_dd x N_chan x 4` Byes. If reading 2^20 (~1 million) channels per gulp, 4 GB of memory is required to search up to 10 Hz/s with 0.01 Hz/s resolution. 

#### Stepped plan

The stepped plan is motivated primarily to reduce memory usage. In the stepped plan, the step between drift rates doubles every `T` trials, where `T` is the number of timesteps. This doubling matches up with dedoppler smearing across channels, so works well in tandem with `apply_smearing_corr`. 
 
The stepped plan is kinda similar to searching your data up to `T` drift trials, then iterative decimating the input data by a factor of 2 (aka 'frequency scrunching') and running the search again at higher drift rates. The number of trials needed is decreased by `O(log2(T))`.

[!optimal-vs-stepped](https://user-images.githubusercontent.com/713251/227719265-2553f687-aebc-4c83-a923-5e304fec601b.png)

### Comparison to turboSETI

Hyperseti uses a brute force dedoppler algorithm, whereas [turboSETI](https://github.com/UCBerkeleySETI/turbo_seti/) uses a computationally efficient [Taylor tree](https://ui.adsabs.harvard.edu/abs/1974A&AS...15..367T). While 'computationally efficient' sounds enticing, the Taylor tree has two drawbacks:

1) The Taylor tree only searches up drifts corresponding to `T` channels, where T is the number of timesteps in the spectrogram. To get to higher driftrates, you need to copy the data, apply a manual
dedoppler correction, and then run the Taylor tree algorithm again. This means many memory operations and can easily wipe out any computational savings (particularly for high drift rates). 
2) The Taylor tree loses reuses precomputed sums to lower its computational cost, and these sums are not
optimal for all drift rates. As a result, not all power is recovered for many drift rates; as the number of timesteps increases the 'lost' power fraction increases [JLM+21](https://ui.adsabs.harvard.edu/abs/2021AJ....161...55M/abstract).

GPUs are particularly good at brute-force algorithms, as they can parallelize across drift-rate trials. The hyperseti dedoppler kernel is very fast, and we argue that other bottlenecks such as disk I/O and peak finding algorithms are more deserving of optimization efforts. 
