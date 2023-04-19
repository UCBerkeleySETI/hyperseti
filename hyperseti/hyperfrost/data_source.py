from copy import deepcopy
from .bifrost import to_bf
import bifrost.pipeline as bfp

from hyperseti.data_array import DataArray
from hyperseti.io import load_data
from hyperseti.hyperfrost.bifrost import gen_bf_metadata
from hyperseti.log import get_logger

logger = get_logger('hyperfrost.data_source')

import numpy as np

class DataArrayReader(object):
    """ Context-managed DataArray reader class for bifrost
    
    Provides context management __enter__ and __exit__. 
    After __init__ provides one single method:
        read() - Read a frame of data
    """
    def __init__(self, filename: str, gulp_size: int, axis: str, overlap: int=0, counter: int=0):
        """ Initialize DataArrayReader to stream data from file-like object
        
        Args:
            filename (str): Name of file
            counter (int): Counter value (used to track iterations, and make sure name is unique)
            gulp_size (int): Size of gulp to take on given axis
            axis (str): axis to take gulp from (i.e. make a sub-array) and iterate over
            overlap (int): Overlap factor between gulp
        
        Notes:
            For an array with dims (time, pol, frequency) and shape (16, 1, 524288)
            We may set gulp=32768 and axis='frequency', to produce a stream of frames
            with shape (16, 1, 32768)
        """

        self.data_array = load_data(filename)
        self.gulp_size = gulp_size
        iter_dict = {axis: gulp_size}
        iter_dict_overlap = {axis: overlap}
        self.data_iterator = self.data_array.iterate_through_data(iter_dict, overlap=iter_dict_overlap)
        self.axis = axis
        self.overlap = overlap
        self.hdr = gen_bf_metadata(self.data_array)
        
        # Update tensor with gulp size
        axis_idx = self.hdr['_tensor']['labels'].index(axis)
        self.hdr['_tensor']['shape'][axis_idx] = gulp_size
        
        # Names across sequences must be unique -- appending counter ensures this
        self.hdr['name'] = f"{counter}_{self.hdr['name']}"
    
    def read(self):
        """ Read a frame of data from iterator"""
        try:
            d =  next(self.data_iterator)
            #print(d.shape)
            return d
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
    def __init__(self, filenames, gulp_size,  axis, overlap=0,  *args, **kwargs):
        gulp_nframe = 1
        if 'gulp_nframe' in kwargs: kwargs.pop('gulp_nframe') # Hardcode to 1
        super().__init__(filenames, gulp_nframe, *args, **kwargs)
        self.gulp_size = gulp_size
        self.axis = axis
        self.overlap = overlap
        self.counter = 1

    def create_reader(self, filename: str):
        """ Uses hyperseti's DataArray class to load data """
        logger.info(f"Loading {filename}")
        self.counter += 1
        return DataArrayReader(filename, self.gulp_size, self.axis, self.overlap, self.counter)
        
    def on_sequence(self, ireader: DataArrayReader, filename: str):
        """ Automatically called whenever a new reader is created (i.e. new file opened) """
        ohdr = deepcopy(ireader.hdr)
        logger.info(ohdr)
        print(type(ireader))
        return [ohdr]

    def on_data(self, reader: DataArrayReader, ospans: list):
        """ Called iteratively, pushes data into output ring """
        data_array = reader.read()
        n_frame = 0
        if isinstance(data_array, DataArray):
            logger.debug(data_array.slice_info[2])
            indata = data_array.data[:]
            gidx = data_array.dims.index(self.axis)

            n_frame = 1
            if indata.shape[gidx] == self.gulp_size: 
                ospans[0].data[0] = indata
            else:
                # Final gulp may not be same size, need to pad to keep memsize the same
                if indata.shape[gidx] > 0:
                    logger.info("Padding array...")
                    pad_shape  =  list(ospans[0].data.shape)[1:]
                    pad_shape[gidx] = pad_shape[gidx] - indata.shape[gidx]
                    padded_array = np.zeros(shape=pad_shape, dtype=indata.dtype)
                    indata_padded = np.concatenate((indata, padded_array), axis=gidx)
                    logger.debug(indata.shape, padded_array.shape, indata_padded.shape)
                    ospans[0].data[0] = indata_padded
                else:
                    n_frame = 0                
        return [n_frame]