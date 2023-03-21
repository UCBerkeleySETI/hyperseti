from hyperseti.hit_browser import HitBrowser
from hyperseti.io.hit_db import HitDatabase
from hyperseti.io import from_h5
import pylab as plt
import cupy as cp
import pandas as pd
import os

try:
    from .file_defs import synthetic_fil, test_fig_dir, voyager_h5, voyager_csv
except:
    from file_defs import synthetic_fil, test_fig_dir, voyager_h5, voyager_csv

def test_browser():
    try:
        darr = from_h5(voyager_h5)
        v = pd.read_csv(voyager_csv)
        hb = HitBrowser(darr, v)

        hb.view_hit(0, 100, plot='waterfall', 
                    plot_config={'waterfall': {'xaxis': 'frequency'}})
        hb.view_hit(2, 64, plot='dedoppler')
        hb.view_hit(0, 64, plot='ddsk')
        hb.view_hit(1, 128, plot='dual')

        hb.to_db('test_browser.hitdb', 'test_obs')

    finally:
        if os.path.exists('test_browser.hitdb'):
            os.remove('test_browser.hitdb')

if __name__ == "__main__":
    test_browser()
