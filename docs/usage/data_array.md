## Working with DataArrays

The central data structure used in hyperseti is the `DataArray`. It is returned when loading data (fil or h5), and internally. For example:

```python
from hyperseti.data_array import from_h5
darr = from_h5('voyager_data.h5')
```
If you inspect this in jupyter you will see:

<img width="650" alt="image" src="https://user-images.githubusercontent.com/713251/126744321-575ecfc6-9f37-4cff-82b1-8e4351c1eaf9.png">

DataArray is similar to [xarray](http://xarray.pydata.org/en/stable/), in that it labels a numpy-like array with dimensions, scales and attribute metadata.
The issue with xarray is that for very large arrays, the coords that describe the axis are also very large (see [pydata:discussion#5166](https://github.com/pydata/xarray/discussions/5156)). For example:
```python
import xarray as xr
data = np.random.rand(1000000, 3)
frequency = np.linspace(1, 2, 1000000)
locs = ['a', 'b', 'c']
xdata = xr.DataArray(data, coords=[frequency, locs], dims=["frequency", "space"])
```
Here, we needed to generate a very large frequency array -- this is slow to create, and uses a lot of memory. For high-time or high-frequency resolution radio data,
this is problematic. 

### Introducing `DimensionScale`

Hyperseti's solution is a class `DimensionScale`, which attaches to the `DataArray` to describe each axis. 
A `DimensionScale` pretends it is a numpy array but is actually just composed of three values: start, stop, and step, that is:
```python
dim_scale_value = start_value + step_size * i
```

It also has units (e.g. GHz) and a name (e.g. 'frequency'):
```python
d = np.arange(2**20)
ds = DimensionScale('frequency', 1.1, 1.9, len(d), 'GHz')
  >> <DimensionScale 'frequency': start 1.1 GHz step 1.9 GHz nstep 1048576 >
```

Dimension scales can be indexed, and a new dimension scale will be generated:
```
ds[1024:1032:2]
  >> <DimensionScale 'frequency': start 1946.6999999999998 GHz step 3.8 GHz nstep 4 >
```

And they can be converted into numpy arrays, or into astropy.Quantity datasets:
```python
# generate numpy array
ds_array = np.asarray(ds)
# generate astropy.Quantity array
ds_astropy = ds.generate()
```

### Parts of the `DataArray`

To construct a DataArray, you need to supply data, dims, scales, and attrs. Here's how to initialize a new array:
```python
darr = DataArray(data, dims, scales, attrs, slice_info=None, parent_shape=None)
```
These correspond to
* `data` - A numpy-like dataset. This can be a numpy.ndarray, a cupy.ndarray, a h5py.Dataset, or anything else that is numpy-like.
* `dims` - The names of each axis of the `data` array, e.g. (frequency, time, polarization).
* `scales` - A set of `DimensionScales`, one for each dimension in `dims`.
* `attrs` - A dictionary of any other metadata you'd like to attach.

The `slice_info` and `parent_shape` are to do with if you have selected a subsection of data from a larger array, so you can keep track. 
These will populate if you call the `isel()` method:

```python
darr = from_h5('voyager_data.h5')
dsel = darr.sel({'frequency': slice(0, 4096, 2), 'time': slice(1, 7)})
```

Which returns a new DataArray:

<img width="650" alt="image" src="https://user-images.githubusercontent.com/713251/126745394-373cf705-ae94-48ca-ae6e-ab17d7d97075.png">
