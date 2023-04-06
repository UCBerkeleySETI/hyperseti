
import os
import re
import pprint
from pathlib import Path
import h5py
import hdf5plugin
import pandas as pd
import yaml

# Imports for metadata creation
import socket
import os
from datetime import datetime
from astropy.time import Time
import sys
from ..version import HYPERSETI_VERSION
from ..hit_browser import HitBrowser
from ..data_array import DataArray
from hyperseti.io.load_data import load_data
from hyperseti.io.load_config import load_config

# READ SCHEMA INFO
HERE = Path(__file__).parent.absolute()
SCHEMA_YML_PATH = os.path.join(HERE, 'hit_db_schema.yml')
SCHEMA_DICT = load_config(SCHEMA_YML_PATH)

#logging
from ..log import get_logger
logger = get_logger('hyperseti.io.hit_db')

def generate_metadata(input_filename: str=None, input_filepath: str=None) -> dict:
    """ Create basic metadata for data progeny 
    
    Args:
        input_filename (str): Name of input data file (optional)
        input_filepath (str): Path to input data file
    """
    username = os.getlogin()
    host     = socket.gethostname()
    user_dict = {
        'username': username, 
        'host': host,
        'python_version': sys.version,
        'hyperseti_version': HYPERSETI_VERSION,
        'workdir': os.getcwd()
        }
    if input_filename is not None:
        user_dict['input_filename'] = input_filename
    if input_filepath is not None:
        user_dict['input_filepath'] = input_filepath

    return user_dict

def get_col_schema(name: str) -> dict:
    """ Read column information from schema 

    Args:
        name (str): Name of column to lookup

    Notes:
        This handles regex lookup for 'bXX_colname_cYY'
    
    Returns:
        col_info (dict): Information about column, as read from schema
    """

    # Regex: does col start with b0_ ... bX_
    pat_beam = r'b(\d+)_(\w+)'
    beam_match = re.search(pat_beam, name)
    if beam_match:
        beam_id = beam_match.group(1)
        col_name = beam_match.group(2)
        
        # Regex: check if it is a poly coefficient (endswith _c0 ... _cX)
        pat_coeff = r'(\w+)_c(\d+)$'
        coeff_match = re.search(pat_coeff, beam_match.group(2))
        if coeff_match:
            col_name = coeff_match.group(1)
            coeff_id = int(coeff_match.group(2))
            
            d = SCHEMA_DICT[f'b{0}_{col_name}_c{0}']
            # Replace 0 indexes with actual index values
            d['description'] = d['description'].replace('coefficient 0', f'coefficient {coeff_id}')
            d['description'] = d['description'].replace('(c0)', f'(c{coeff_id})')
            d['description'] = d['description'].replace('(c0)', f'(c{coeff_id})')
            d['description'] = d['description'].replace('beam 0', f'beam {beam_id}')
            d['description'] = d['description'].replace('(b0)', f'(b{beam_id})')
        else:
            d = SCHEMA_DICT[f'b{0}_{col_name}']
            # Replace 0 indexes with actual index values
            d['description'] = d['description'].replace('beam 0', f'beam {beam_id}')
            d['description'] = d['description'].replace('(b0)', f'(b{beam_id})')
    else:
        d = SCHEMA_DICT[name]
    return d



class HitDatabase(object):
    """ Simple HDF5-backed database for storing hits from multiple observations
    
    HitDatabase provides:
        hit_db.list_obs() - List all observations within the database file.
        hit_db.add_obs()  - Add a new observation to database.
        hit_db.get_obs()  - Retrieve hit data table from the database.
    
    Notes:
        This is intended for storing results from a hyperseti run on multiple files.
        It is not intended as a long-term SQL-style hit database. 
        This is currently basic, and API may change so that more information can
        be stored (e.g. logs, config, and other things for reproducability)
    
    Data model notes:
        Within the HDF5 file, a new group is created in the root for each observation.
        Table data are a column store, one dataset per column, i.e.:
        
        ```
        GROUP "/" {
            GROUP "proxima_cen" {
                DATASET "beam_idx"
                DATASET "boxcar_size"
                DATASET "snr"
                DATASET ...}
            GROUP "alpha_cen" {
                DATASET "beam_idx"
                DATASET "boxcar_size"
                DATASET "snr"
                DATASET ...}
            GROUP ... {
                DATASET ...
            }  
        }
        ```
    """
    def __init__(self, filename: str, mode: str='r'):
        """ Initialize HitDatabase object

        Args:
            filename (str): Name of database file to load
            mode (str): 'r' for readonly or 'w' for write
        """
        self.filename = filename
        self.h5 = h5py.File(filename, mode=mode)
        if mode == 'w':
            self.h5.attrs['CLASS']   = 'HYPERSETI_DB'
            self.h5.attrs['VERSION'] = HYPERSETI_VERSION

    def __del__(self):
        self.h5.close()
    
    def close(self):
        self.__del__()
    
    def __repr__(self):
        return f"< HitDatabase: {self.filename} N_obs: {len(list(self.h5.keys()))} >"

    def list_obs(self) -> list:
        """ List all observations in the file

        Returns:
            obs_list (list): List of all observations (h5 groups) in file.
        """
        return list(self.h5.keys())
    
    def add_obs(self, obs_id: str, hit_table: pd.DataFrame, input_filename: str=None, config: dict=None):
        """ Add a new observation hit table to the database 
        
        Args:
            obs_id (str): Unique name for observation (e.g. input filterbank filename)
            hit_table (pd.DataFrame): DataFrame to store
            input_filename (str): Filename of input
            config (dict): Pipeline config to add (Adds obs_group.attrs['pipeline_config'] as YAML dump)

        Notes:
            Need to open file with mode='w'
        """
        obs_group = self.h5.create_group(obs_id)
        for key in hit_table.columns:
            try:
                col_md = get_col_schema(key)
            except KeyError:
                logger.warning(f"Column ID {key} not in schema.")
                col_md = {'dtype': None}    # WAR to ensure create_dataset works

            dset = obs_group.create_dataset(key, data=hit_table[key], 
                                            dtype=col_md['dtype'])

            # Add other schema metadata
            for md in ('description', 'units'):
                if col_md.get(md, None):
                    dset.attrs[md] = col_md[md]
        
        # Generate metadata
        for key, val in generate_metadata().items():
            obs_group.attrs[key] = val
        
        if config is not None:
            obs_group.attrs['pipeline_config'] = yaml.dump(config)
        
        if input_filename is not None:
            obs_group.attrs['input_filepath'] = os.path.dirname(input_filename)
            obs_group.attrs['input_filename'] = os.path.basename(input_filename)
    
    def get_obs(self,  obs_id: str) -> pd.DataFrame:
        """ Retrieve a DataFrame from the HDF5 database 

        Args:
            obs_id (str): Name of observation to retrieve
        
        Returns:
            hit_table (pd.DataFrame): Pandas DataFrame of hits
        """
        obs_group = self.h5[obs_id]
        obs_dict = {}
        for key, dset in obs_group.items():
            obs_dict[key] = dset[:]
        return pd.DataFrame(obs_dict)
    
    def get_obs_metadata(self,  obs_id: str) -> dict:
        """ Retrieve metadata information from the HDF5 database 

        Args:
            obs_id (str): Name of observation metadata to retrieve
        
        Returns:
            md (dict): Metadata dictionary (generated by generate_metadata())
        """
        d = {}
        for key, val in self.h5[obs_id].attrs.items():
            d[key] = val
        return d

    def get_obs_schema(self,  obs_id: str) -> dict:
        """ Retrieve schema information from the HDF5 database 

        Args:
            obs_id (str): Name of observation metadata to retrieve
        
        Returns:
            schema (dict): Metadata dictionary
        """
        d = {}
        for dset_name, dset in self.h5[obs_id].items():
            d[dset_name] = dict(dset.attrs)
            d[dset_name]['dtype'] = dset.dtype
        return d

    def get_obs_config(self,  obs_id: str) -> dict:
        """ Retrieve pipeline config information from the HDF5 database 

        Args:
            obs_id (str): Name of observation metadata to retrieve
        
        Returns:
            md (dict): Metadata dictionary (generated from stored YAML, if present)
        """
        return yaml.load(self.h5[obs_id].attrs['pipeline_config'], yaml.Loader)  

    def browse_obs(self, obs_id: str, data_array: DataArray=None) -> HitBrowser:
        """ Return a HitBrowser object for given obs_id

        Args:
            obs_id (str): Name of observation metadata to retrieve
            data_array (str): If set, instead of attempting to use 
                              the file metadata within the database,
                              use the provided DataArray object.
        
        Returns:
            hb (HitBrowser): HitBrowser object for given observation

        Notes:
            Requires that the file name and path listed in the metadata
            are accurate. A FileNotFoundError will be raised otherwise
        """
        md = self.get_obs_metadata(obs_id)
        obs_group = self.h5[obs_id]
        
        if isinstance(data_array, DataArray):
            hit_table = self.get_obs(obs_id)
            return HitBrowser(data_array, hit_table)
        if 'input_filepath' in md.keys() and 'input_filename' in md.keys():
            fn = os.path.join(md['input_filepath'], md['input_filename'])
            if os.path.exists(fn):
                data_array = load_data(fn)
                hit_table = self.get_obs(obs_id)
                return HitBrowser(data_array, hit_table)
            else:
                raise FileNotFoundError("Cannot find original data. Please use data_arr= argument to manually link.")
        else:
            raise FileNotFoundError("Cannot find original data. Please use data_arr= argument to manually link.")




