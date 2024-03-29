from .hdf5 import from_h5
from .filterbank import from_fil
from .setigen import from_setigen
from ..data_array import from_metadata, DataArray
from hyperseti.thirdparty import sigproc

import h5py

def load_data(filename: str) -> DataArray:
    """ Load file and return data array

    Args:
        filename (str): Name of file to load. 
    
    Returns:
        ds (DataArray): DataArray object
    
    Notes:
        This method also supports input as a setigen Frame,
        or as an existing DataArray.
    """

    if isinstance(filename, DataArray):
        ds = filename           # User has actually supplied a DataArray
    elif isinstance(filename, str):
        if h5py.is_hdf5(filename):
            ds = from_h5(filename)
        elif sigproc.is_filterbank(filename):
            ds = from_fil(filename)
        else:
            raise RuntimeError("Only HDF5/filterbank files or DataArray/setigen.Frame objects are currently supported")
    else:
        #WAR: setigen import is slow, only do if necessary.
        from setigen import Frame 
        if isinstance(filename, Frame):
            ds = from_setigen(filename)
        else:
            raise RuntimeError("Only HDF5/filterbank files or DataArray/setigen.Frame objects are currently supported")
    return ds