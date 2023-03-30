from __future__ import annotations  # Treat type hints as strings to avoid circular import

import pandas as pd
import cupy as cp
import numpy as np
import pylab as plt
from astropy import units as u

from hyperseti.plotting import imshow_dedopp, imshow_waterfall, overlay_hits
from hyperseti.data_array import DataArray
from hyperseti.dedoppler import dedoppler, calc_delta_dd


class HitBrowser(object):
    """ Class for browsing hits after running pipeline 
    
    HitBrowser provides:
        browser = HitBrowser(data_array, hit_table)
        browser.extract_hit() - extract a hit from the data_array and return data
        browser.view_hit() - plot a hit from the hit table
        browser.to_db() - write to database

    """
    def __init__(self, data_array: DataArray, hit_table: pd.DataFrame):
        """ Initialize HitBrowser 

        Args:
            data_array (DataArray): Input DataArray
            hit_table (pd.DataFrame): DataFrame of hits found by pipeline
        """
        self.data_array = data_array
        self.hit_table  = hit_table
    
    def __repr__(self):
        return f"< HitBrowser: {self.data_array.attrs['source']} N_hits: {len(self.hit_table)} >"

    def to_db(self, db: HitDatabase, obs_id: str):
        """ Write to database 
        
        Args:
            db (HitDatabase or str): HitDatabase object, OR a filename string.
            obs_id (str): Name of observation within HitDatabase
        """
        if isinstance(db, str):
            from hyperseti.io import hit_db   # Avoid circular import
            db = hit_db.HitDatabase(db, mode='w')
        db.add_obs(obs_id, self.hit_table, input_filename=self.data_array._filename)
    
    def extract_hit(self, hit_idx: int, padding: int, space='cpu') -> DataArray:
        """ Extract a hit from the data_array 
        
        Args:
            hit_idx (int): ID of hit to plot (DataFrame iloc)
            padding (int): Number of channels to pad around hit
            space (str): Space where data is loaded, either 'cpu' or 'gpu'
        
        Returns:
            data_sel (DataFrame): Extracted DataArray object 
        """
        hit = self.hit_table.iloc[hit_idx]

        t_elapsed = self.data_array.time.elapsed.to('s').value
        f_step    = self.data_array.frequency.step.to('Hz').value

        chan_idx = int(hit['channel_idx'])
        beam_idx   = int(hit['beam_idx'])
        drift_rate = hit['drift_rate']
        n_chan_drift = int(drift_rate * t_elapsed) // 2
        n_box        = int(hit['boxcar_size']) // 2

        
        chan0 = chan_idx - padding - n_box
        chanX = chan_idx + padding + n_box 

        if n_chan_drift * f_step < 0: 
            chan0 -= abs(n_chan_drift) 
        else:
            chanX += abs(n_chan_drift) 

        data_sel = self.data_array.sel(
            {'frequency': slice(chan0, chanX)}
            )
        if space == 'cpu':
            data_sel.data = cp.asnumpy(data_sel.data)
        else:
            data_sel.data = cp.asarray(data_sel.data)
        return data_sel    

    def _overlay_hit(self, hit_idx):
        hit = self.hit_table.iloc[hit_idx].copy()
        hit['channel_idx'] = 0              # Set to zero as we are plotting offsets
        overlay_hits(hit)
    
    def view_hit(self, hit_idx:int, padding: int=128, 
                 plot: str='waterfall', overlay_hit: bool=False, plot_config: dict={}):
        """ Plot hits in database (postage stamp style)

        Args:
            hit_idx (int): ID of hit to plot (DataFrame iloc)
            padding (int): Number of channels to pad around hit
            plot (str): Plot type, one of 'waterfall', 'dedoppler' or 'dual'
            overlay_hit (bool): Overlay an x marker where the hit was found
            plot_config (dict): Plotting kwargs to pass on to imshow functions
        
        Notes:
            Config dict is passed on to hyperseti.plotting.imshow functions,
            here's and example:
            ```
            plot_config = {'waterfall': {'xaxis': 'channel'}, 
                           'dedoppler': {'show_colorbar': False}}
            ```
        """
        
        hit_darr = self.extract_hit(hit_idx, padding, space='cpu')
        obs_len, delta_dd = calc_delta_dd(hit_darr)
                    
        if plot not in ('waterfall', 'dedoppler', 'ddsk', 'dual'):
            raise RuntimeError("plot arg not understood, choose waterfall, dedoppler or dual.")

        if plot in ('dedoppler', 'dual', 'ddsk'):
            N_DD_MIN = 32                               # Minimum number of dedoppler bins in dedoppler plot

            DD_MIN = N_DD_MIN * delta_dd                # Compute minimum max_dd for plot
            max_dd = abs(self.hit_table.iloc[hit_idx]['drift_rate'])
            max_dd *= 2
            max_dd = np.max([max_dd, DD_MIN])

            # Reload data so we can pad with maximum dd
            n_chan_dedopp = abs(int(max_dd / delta_dd))
            hit_darr = self.extract_hit(hit_idx, padding + n_chan_dedopp, space='cpu')
            hit_darr.data = cp.asarray(hit_darr.data)
            if plot == 'ddsk':
                hit_dedopp, hit_dedopp_sk = dedoppler(hit_darr, max_dd=max_dd, 
                plan='optimal', kernel='ddsk')
                hit_dedopp_sk.data = cp.asnumpy(hit_dedopp_sk.data)
            else:
                hit_dedopp = dedoppler(hit_darr, max_dd=max_dd, 
                plan='optimal', kernel='dedoppler')
            hit_dedopp.data = cp.asnumpy(hit_dedopp.data)

        if plot == 'waterfall':
            kwargs = plot_config.get('waterfall', {})
            imshow_waterfall(hit_darr, **kwargs)
        
        elif plot == 'dedoppler':
            kwargs = plot_config.get('dedoppler', {})
            imshow_dedopp(hit_dedopp, **kwargs)

        elif plot == 'ddsk':
            kwargs = plot_config.get('ddsk', {})
            imshow_dedopp(hit_dedopp_sk, **kwargs)

        elif plot == 'dual':
            plt.subplot(1,2,1)
            kwargs = plot_config.get('waterfall', {})
            imshow_waterfall(hit_darr, **kwargs)
            plt.subplot(1,2,2)
            kwargs = plot_config.get('dedoppler', {})
            imshow_dedopp(hit_dedopp, **kwargs)
        
        if overlay_hit:
            self._overlay_hit(hit_idx)

        




