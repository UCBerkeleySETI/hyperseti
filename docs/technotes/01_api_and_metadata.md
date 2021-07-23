## API approach

Key functions are written with the first two arguments:
  ```python
  def dedoppler(data, metadata, ...):
  ```
Where data is a numpy-like array, and metadata is a python dictionary of metadata needed to understand the data array.

Where possible, key functions should also return `(data, metadata)` as their first two items. 

These functions are wrapped with a decorator `@datwrapper()` ([code here](https://github.com/UCBerkeleySETI/hyperseti/blob/master/hyperseti/utils.py#L86)), 
which allows a `DataArray` to be used instead of a bare
numpy array + metadata dictionary.  `@datwrapper` supplies the `metadata=` kwarg to wrapped function, which it derives
from attributes of the DataArray. It also splits off the DataArray.data and returns that as first argument.

The `@datwrapper` has a `dims=` argument, which if supplied will take the output returned by the wrapped function and 
convert it into a `DataArray` with the supplied dims. For example

```python
@datwrapper(dims=('drift_rate', 'beam_id', 'frequency'))
def dedoppler(data, metadata, ...'):
```

Means that the output will be a `DataArray` with dimensions (drift_rate, beam_id, frequency). 
If the function does *not* return (metadata, data) as its first two arguments, `@datwrapper(dims=None)` must be set.

### Why do this?

This is an attempt to allow the user to supply either (data, metadata), or to use a `DataArray` object. For testing,
and in earlier versions, supplying data/metadata pair was easier. This means within the function, we only have to deal
with a numpy-like array and a metadata dictionary, which I hope is a little more readable. It is also quite nice to be
able to see just above the function def what the output dims will be.

### Current issues to fix 
Limitations and annoying things about the `@datwrapper` approach:
* If you're not careful, and you do something like `data_dedoppler, metadata = dedoppler(data, metadata)`, you can accidentally confuse yourself and use the wrong metadata dict.
* ~~Currently the datwrapper either doesn't return a `DataArray`, or needs to be told the dims of the new `DataArray`. It would be nice to allow it to return a data array with original dimensions (e.g. normalize and apply_boxcar functions do not change dimensions).~~ *Added 23 Jul 21*
* Hard requrement that the wrapped function has `(data, metadata)` as first two arguments. 
* If you supply `data=DataArray` AND ALSO `metadata=Dict`, you get an error raised.
* ~~If you supply a `DataArray`, it can cause unexpected errors with proceeding arguments.~~ *Fixed 24 Jul 21*

## Metadata approach

The metadata required currently varies from function to function, but is based on dimension scales. They should have the 
form `[scalename]_start` and `[scalename]_stop`. E.g.

```python
metadata = {
  'frequency_start': 1 * u.GHz,
  'frequency_step':  1 * u.Hz,
  'time_start':  Time(),
  'time_step': 1 * u.s
  }
```
