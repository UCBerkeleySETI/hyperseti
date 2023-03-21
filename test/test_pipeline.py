import pylab as plt
import cupy as cp
import numpy as np

from hyperseti.io import from_fil, from_h5
from hyperseti.pipeline import logger
import logbook
logger.level = logbook.WARNING

try:
    from .file_defs import synthetic_fil, test_fig_dir, voyager_h5
except:
    from file_defs import synthetic_fil, test_fig_dir, voyager_h5

from hyperseti.pipeline import GulpPipeline, find_et

config = {
    'preprocess': {
        'sk_flag': True,
        'normalize': True,
        'poly_fit': 5,
        'blank_edges': {'n_chan': 32768},
        'blank_extrema': {'threshold': 10000},
    },
    'dedoppler': {
        'kernel': 'dedoppler',
        'max_dd': 4.0,
        'min_dd': None,
        'apply_smearing_corr': False,
        'plan': 'stepped',
        },
    'hitsearch': {
        'threshold': 20,
        'min_fdistance': 500,
        'merge_boxcar_trials': True
    },
    'pipeline': {
        'n_boxcar': 9,
        'n_blank': 2,
    }
}

def test_pipeline():
    print("--- Stepped ---")
    hits_et = find_et(voyager_h5, config, gulp_size=2**20)
    print(hits_et)

    print("--- Stepped ---")
    config['dedoppler']['plan']   = 'optimal'
    hits_et = find_et(voyager_h5, config, gulp_size=2**20)
    print(hits_et)

    print("--- n_boxcar = 1 ---")
    config['dedoppler']['plan']   = 'stepped'
    config['dedoppler']['apply_smearing_corr'] = False
    config['pipeline']['n_boxcar'] = 1
    hits_et = find_et(voyager_h5, config, gulp_size=2**20)
    print(hits_et)

    print("--- smearing_corr on ---")
    config['dedoppler']['apply_smearing_corr'] = True
    hits_et = find_et(voyager_h5, config, gulp_size=2**20)
    print(hits_et)

def test_pipeline_to_db():
    config['pipeline']['n_boxcar'] = 1
    hits_et = find_et(voyager_h5, 
                        config, 
                        gulp_size=2**20,
                        filename_out='test_voyager.hitdb'
                        )
    print(hits_et)   


if __name__ == "__main__":
    #test_pipeline()
    test_pipeline_to_db()