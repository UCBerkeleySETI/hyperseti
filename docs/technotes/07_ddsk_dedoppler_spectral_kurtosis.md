## DDSK Kernel

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

**Research question: can we learn anything about the ISM from ddsk measurements of Voyager / Mars missions?**