## Conda recipes

Recipe for building conda package. From the repository root directory, run:

```
conda-build conda-recipe -c conda-forge
```

Upload to [anaconda/technosignatures](https://anaconda.org/technosignatures/hyperseti) channel:

```
anaconda login
anaconda upload --user technosignatures /home/dancpr/conda-bld/linux-64/hyperseti-1.0.0-0.tar.bz2
```


