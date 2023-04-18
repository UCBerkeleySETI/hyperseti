from copy import deepcopy
from .bifrost import to_bf
import bifrost.pipeline as bfp

from hyperseti.data_array import DataArray
from hyperseti.io import load_data
from hyperseti.hyperfrost.bifrost import gen_bf_metadata

class DataArrayReader(object):

    def __init__(self, filename, gulp_size, axis, overlap=0):

        self.data_array = load_data(filename)
        self.gulp_size = gulp_size
        iter_dict = {axis: gulp_size}
        print(iter_dict)
        self.data_iterator = self.data_array.iterate_through_data(iter_dict)
        self.axis = axis
        self.overlap = overlap
        self.hdr = gen_bf_metadata(self.data_array)
        
        # Update tensor with gulp size
        axis_idx = self.hdr['_tensor']['labels'].index(axis)
        self.hdr['_tensor']['shape'][axis_idx] = gulp_size
    
    def read(self):
        try:
            return next(self.data_iterator)
        except StopIteration:
            return [0]
        
    def __enter__(self):
        return self

    def close(self):
        self.data_array.close()

    def __exit__(self, type, value, tb):
        self.close()

class DataArrayBlock(bfp.SourceBlock):
    """ Block for reading binary data from file and streaming it into a bifrost pipeline

    Args:
        filenames (list): A list of filenames to open
        gulp_size (int): Number of elements in a gulp (i.e. sub-array size)
        axis (str): name of axis to take gulps from
    """
    def __init__(self, filenames, gulp_size, gulp_nframe, axis, overlap=0,  *args, **kwargs):
        super().__init__(filenames, gulp_nframe, *args, **kwargs)
        self.gulp_size = gulp_size
        self.axis = axis
        self.overlap = overlap

    def create_reader(self, filename):
        print(f"Loading {filename}")
        return DataArrayReader(filename, self.gulp_size, self.axis, self.overlap)

    def on_sequence(self, ireader, filename):
        ohdr = deepcopy(ireader.hdr)
        print(ohdr)
        return [ohdr]

    def on_data(self, reader, ospans):
        data_array = reader.read()
        if isinstance(data_array, DataArray):
            print(data_array.slice_info)
            indata = data_array.data[:]
            if indata.shape[2] == self.gulp_size: # idx 2 == frequency
                ospans[0].data[0] = indata
                return [1]
            else:
                return [0]
        else:
            return [0]