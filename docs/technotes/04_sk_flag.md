## Spectral Kurtosis flagging

[Kurtosis](https://en.wikipedia.org/wiki/Kurtosis) is essentially a statistical measure of how un-noiselike data is.

In the context of radio astronomy, 'spectral kurtosis' (or SK) is applying kurtosis ideas to waterfall plots,
computing statistics for each frequency channel. This is laid out in [Nita & Dale (2007)](https://ui.adsabs.harvard.edu/abs/2007PASP..119..805N/abstract). The [SK estimator](https://ui.adsabs.harvard.edu/abs/2010MNRAS.406L..60N/abstract) 
is an easy way to calculate spectral kurtosis in the case that some time averaging has occured.

In short, the time stream for a channel in a power spectral density waterfall plot will:
1) be a chi-squared distribution if it follows radiometer noise. In this case spectral kurtosis value is approx equal to 1.
2) have Kurtosis close to zero if dominated a constant tone.
3) Kurtosis will be large if dominated by impulsive interference.

Most commonly, SK is used in high time resolution data to get rid of impulsive interference.

### Using SK for narrowband tone detection
To use SK to detect constant wave tones we need to look for values close to zero. However, before we can use an
N-sigma threshold we need to first normalize the data by taking the log:

![SK-log-for-memo](https://user-images.githubusercontent.com/713251/164014965-d9cc2009-8989-4395-8e74-c97a6fd1dc54.png)

The data here are from the Voyager example, so there are some drifting tones and a DC bin. The drifting tones *cross* over frequency bins, so each bin registers a high kurtosis value. If the
tone had zero drift, in contrast, that frequency bin would have a near-zero kurtosis value.

TLDR: Any frequency bins including constant tones (that don't drift) will have a low SK value. 
Bins that are crossed by drifting tones will have high SK values. Bins with only noise will have
SK values close to 1.

### SK definition

```python
    SK =  ((N_acc × n) + 1) / (n-1) * (n (∑x²) / (∑x)²) - 1)
```