## Dedoppler

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

A bit of real-world testing will be important here to figure out when it's useful. But the following should hold:
* 'drifting squiggles' due to unstable LOs will have high ddsk.
* local bright impulsive RFI will have high ddsk.
* stable drifting tones will have ddsk < 1
* really low values of ddsk probably mean it's local RFI (we would expect some scintillation?).
