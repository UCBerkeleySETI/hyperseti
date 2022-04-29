## Current state of hyperseti, April 2022

hyperseti 0.0.6 is now installed on the blpc nodes. To access it:

```
source /opt/conda/init.sh
conda activate hyperseti
```

There is a findET  command line utility (thanks Richard!) which I've so far only tested on the Voyager data to create a CSV file of hits, and a tsdat utility to  convert output into a turbo_seti compatible csv file.

### Nice features:
* Applies spectral kurtosis (SK) flagging before applying normalization, so S/N estimate should be better
* Applies a polynomial fit to the bandpass to account for the edges of the channel during normalization
* Applies a correction for spectral smearing due to dedoppler, retrieving a sqrt(N_smear) improvement in S/N for high drift rates
* Computes the dedoppler corrected spectral kurtosis, henceforth DDSK, along all driftrate paths (at same time as dedoppler is calculated). This can be used to tell the difference between a stable tone (low SK value) and a modulated/unstable tone (high SK value)
* Can run iteratively with different boxcar widths to search for wider band signals
* New iterative blanking approach, which can find hits hiding behind bright hits
* Handy DataArray class, which displays a nice visual summary in jupyter notebooks.
* Can read setigen frames (using from_setigen function)
* smarter logging system using logbook
* some nice plotting for dedoppler space,  including an overlay_hits  method
* lots of tests using setigen and 90% code coverage

### Not nice features:
* It currently uses an unreasonable amount of GPU memory, not quite sure why but my suspicion is that my on_gpu decorator is making numerous copies of the array.
* The datwrapper decorator, while kinda cool, is admittedly a complex solutions that probably made my life harder overall.
* dedoppler is brute-force, not taylor tree. But its' very fast and not the bottleneck in the search pipeline  (the hit search is the bottleneck)
* Not faster than turboseti (hitsearch bringing the team down)
* Currently requires cusignal, which requires conda and is a bit of a pain to install.
* I meant to design in polarization/multibeam support, but pretty sure it's broken.