import pandas as pd
import cupy as cp
import pylab as plt

from hyperseti.plotting import imshow_dedopp, imshow_waterfall
from hyperseti.data_array import DataArray
from hyperseti.dedoppler import dedoppler

class HitBrowser(object):
    """ Class for browsing hits after running pipeline 
    
    HitBrowser provides:
        browser = HitBrowser(data_array, hit_table)
        browser.extract_hit() - extract a hit from the data_array and return data
        browser.view_hit() - plot a hit from the hit table

    """
    def __init__(self, data_array: DataArray, hit_table: pd.DataFrame):
        """ Initialize HitBrowser 

        Args:
            data_array (DataArray): Input DataArray
            hit_table (pd.DataFrame): DataFrame of hits found by pipeline
        """
        self.data_array = data_array
        self.hit_table  = hit_table
    
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
        f_start   = self.data_array.frequency.start

        chan_idx = int(hit['channel_idx'])
        beam_idx   = int(hit['beam_idx'])
        drift_rate = hit['drift_rate']
        n_chan_drift = int(drift_rate * t_elapsed) // 2
        n_box        = int(hit['boxcar_size']) // 2

        
        chan0 = chan_idx - padding - n_box
        chanX = chan_idx + padding + n_box 

        if n_chan_drift * f_step < 0:
            chan0 += abs(n_chan_drift) 
        else:
            chanX += abs(n_chan_drift) 

        data_sel = self.data_array.isel(
            {'frequency': slice(chan0, chanX)}
            )
        if space == 'cpu':
            data_sel.data = cp.asnumpy(data_sel.data)
        else:
            data_sel.data = cp.asarray(data_sel.data)
        return data_sel    

    
    def view_hit(self, hit_idx:int, padding: int=128, 
                 plot: str='waterfall', plot_config: dict={}):
        """ Plot hits in database (postage stamp style)

        Args:
            hit_idx (int): ID of hit to plot (DataFrame iloc)
            padding (int): Number of channels to pad around hit
            plot (str): Plot type, one of 'waterfall', 'dedoppler' or 'dual'
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

        if plot not in ('waterfall', 'dedoppler', 'ddsk', 'dual'):
            raise RuntimeError("plot arg not understood, choose waterfall, dedoppler or dual.")

        if plot in ('dedoppler', 'dual', 'ddsk'):
            max_dd = abs(self.hit_table.iloc[hit_idx]['drift_rate'])
            max_dd *= 2
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




        




