import h5py
import hdf5plugin
import pandas as pd

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

    def __del__(self):
        self.h5.close()
    
    def __repr__(self):
        return f"< HitDatabase: {self.filename} N_obs: {len(list(self.h5.keys()))} >"

    def list_obs(self) -> list:
        """ List all observations in the file

        Returns:
            obs_list (list): List of all observations (h5 groups) in file.
        """
        return list(self.h5.keys())
    
    def add_obs(self, obs_id: str, hit_table: pd.DataFrame):
        """ Add a new observation hit table to the database 
        
        Args:
            obs_id (str): Unique name for observation (e.g. input filterbank filename)
            hit_table (pd.DataFrame): DataFrame to store

        Notes:
            Need to open file with mode='w'
        """
        obs_group = self.h5.create_group(obs_id)
        for key in hit_table.columns:
            obs_group.create_dataset(key, data=hit_table[key])
    
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




