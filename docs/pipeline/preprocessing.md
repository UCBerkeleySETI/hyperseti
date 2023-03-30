## Preprocessing

Preprocessing prepares the data for dedoppler searching.

### Normalization

Normalizing the data refers to converting it into units of signal-to-noise, by applying

```
data =  (data - mean(data)) / stdev(data)
```

When applying normalization, it is important to first flag outlier data points (i.e. radio interference)
which would bias the calculation. A frequency-dependent bandpass correction can also be applied. 

Hyperseti uses spectral kurtosis flagging to find any points which are not noise-like and flag them. 
Note that hyperseti does *not* remove these signals from the data when searching for hits, we just make sure
they are not used when calculating S/N.

### Spectral Kurtosis flagging

[Kurtosis](https://en.wikipedia.org/wiki/Kurtosis) is essentially a statistical measure of how un-noiselike data is.

In the context of radio astronomy, 'spectral kurtosis' (or SK) is applying kurtosis ideas to waterfall plots,
computing statistics for each frequency channel. This is laid out in [Nita & Dale (2007)](https://ui.adsabs.harvard.edu/abs/2007PASP..119..805N/abstract). The [SK estimator](https://ui.adsabs.harvard.edu/abs/2010MNRAS.406L..60N/abstract) 
is an easy way to calculate spectral kurtosis in the case that some time averaging has occured.

In short, the time stream for a channel in a power spectral density waterfall plot will:
1) be a chi-squared distribution if it follows radiometer noise. In this case spectral kurtosis value is approx equal to 1.
2) have Kurtosis close to zero if dominated a constant tone.
3) Kurtosis will be large if dominated by impulsive interference.

Most commonly, SK is used in high time resolution data to get rid of impulsive interference.

#### Using SK for narrowband tone detection
To use SK to detect constant wave tones we need to look for values close to zero. However, before we can use an
N-sigma threshold we need to first normalize the data by taking the log:

![SK-log-for-memo](https://user-images.githubusercontent.com/713251/164014965-d9cc2009-8989-4395-8e74-c97a6fd1dc54.png)

The data here are from the Voyager example, so there are some drifting tones and a DC bin. The drifting tones *cross* over frequency bins, so each bin registers a high kurtosis value. If the
tone had zero drift, in contrast, that frequency bin would have a near-zero kurtosis value.

TLDR: Any frequency bins including constant tones (that don't drift) will have a low SK value. 
Bins that are crossed by drifting tones will have high SK values. Bins with only noise will have
SK values close to 1.

#### Setting sigma

Hyperseti does flagging on the log of SK values, using the following:

```
    std_log  = 2 / sqrt(N_acc)         # Based on setigen
    mean_log = -1.25 / N_acc           # Based on setigen  
    mask  = np.abs(log_sk) > abs(mean_log) + (std_log * n_sigma)
```

Where `N_acc` is the number of timesteps in the dynamic spectrum, and `n_sigma` number of stdev above which to flag,
and (set by the user). Note the mean_log value is not canonical: it was arrived at by a fit to `setigen` noise data. 

#### Why are we using SK flagging?

We are using SK flagging when normalizing data. To normalize data, we want to compute the signal-to-noise ratio. By flagging anything with spurious SK values, we can get a good estimate of the true noise.

#### SK definition

```python
    SK =  ((N_acc × n) + 1) / (n-1) * (n (∑x²) / (∑x)²) - 1)
```

Where `N_acc` is the number of accumulations per time bin, and `n` is the length of the x array.

### Blanking

While the SK flagging does not blank data (i.e. set it to zero), there are two methods that do blank data:

1) `config['preprocessing']['blank_extrema'] = MAX_SNR`: This will blank any stupidly bright signals with a S/N greater than user-supplied `float=MAX_SNR`.
2) `config['preprocessing']['blank_edges'] = N_CHAN`: This blanks the edges of the gulp. Can be useful if the bandpass falls off steeply at the edges. Polynomial fits are also generally rubbish at the band edges which can cause issues if the edges are not flagged.

### Bandpass removal

Hyperseti currently has one method for bandpass removal: polynomial subtraction. Use the `config['preprocessing']['poly_fit'] = N` parameter to apply. 