"""
Try to get results that are similar to those of turbo_seti with our standard Voyager 1 .h5 file.

# ----------------------------------------------------
# File ID: xx.h5 
# ----------------------------------------------------
# Source:Voyager1
# MJD: 57650.782094907408       RA: 17h10m03.984s       DEC: 12d10m58.8s
# DELTAT:  18.253611    DELTAF(Hz):  -2.793968  max_drift_rate:   4.000000      obs_length: 292.057776
# --------------------------
# Top_Hit_#     Drift_Rate      SNR     Uncorrected_Frequency   Corrected_Frequency     Index   freq_start      freq_end        SEFD    SEFD_freq       Coarse_Channel_Number   Full_number_of_hits 
# --------------------------
001      -0.392226      156.379913         8419.319368     8419.319368  64651      8419.321003     8419.317740  0.0           0.000000  0       77634
002      -0.373093      1258.281250        8419.297028     8419.297028  72647      8419.298662     8419.295399  0.0           0.000000  0       77634
003      -0.392226      159.497269         8419.274374     8419.274374  80755      8419.276009     8419.272745  0.0           0.000000  0       77634

DCP NOTE:
    With manual processing, I find a mean noise level of 1.10575503e+10, std of 1.1497335e+09
    After normalization, in first time step (time-frequency space), max = 259.43027, std = 1.0 
    If noise integrates down as sqrt(N_timesteps), we expect S/N to increase by sqrt(16) = 4x
    So should see an SNR of 1000 
"""

import os
import logbook
import numpy as np
from hyperseti import find_et
from hyperseti.log import update_levels, get_logger
from hyperseti.utils import time_logger
from hyperseti.io.hit_db import HitDatabase

from hyperseti.test_data import voyager_h5

# Other parameters:
GULP_SIZE = 1048576
MAX_DRIFT_RATE = 4.0
MIN_DRIFT_RATE = 0.0
SNR_THRESHOLD = 20.0
N_BOXCAR = 2
KERNEL = "dedoppler"
GPU_ID = 0


def test_with_voyager():
    try:
        print("hyperseti find_et from file {} .....".format(voyager_h5))
        update_levels(logbook.WARNING, [])
        #time_logger.level = logbook.INFO

        config = {
            'preprocess': {
                'sk_flag': True,
                'normalize': True,
                'blank_edges': {'n_chan': 32}
            },
            'dedoppler': {
                'kernel': 'ddsk',
                'max_dd': 5.0,
                'min_dd': None,
                'apply_smearing_corr': False, # Use either this OR pipeline/merge_boxcar_trials
                'plan': 'stepped'
            },
            'hitsearch': {
                'threshold': 100,
                'min_fdistance': None
            },
            'pipeline': {
                'n_boxcar': 10,
                'merge_boxcar_trials': True
            }
        }

        hit_browser = find_et(voyager_h5, config, 
                        gulp_size=2**18,  # Note: intentionally smaller than 2**20 to test slice offset
                        filename_out='./test_voyager_hits.csv',
                        log_output=True,
                        log_config=True
                        )
                        
        dframe = hit_browser.hit_table
    
        # dframe column names: drift_rate  f_start  snr  driftrate_idx  channel_idx  boxcar_size  beam_idx  n_integration
        print("Returned dataframe:\n", dframe)
        print(dframe.dtypes)
        list_drate = dframe["drift_rate"].tolist()

        assert os.path.exists('test_voyager_hits.csv')
        assert os.path.exists('test_voyager_hits.yaml')
        assert os.path.exists('test_voyager_hits.log')

        # This is a quick test to check if smaller gulps are taking the channel offset into account
        assert np.alltrue(dframe['channel_idx'] > 739000)

        for drate in list_drate:
            print("Observed drift rate = {}, should be negative.".format(drate))
            assert drate <= 0.0

        # Second iteration -- use blank_hits dict
        config['pipeline']['blank_hits'] = {'n_blank': 2, 'padding': 10}

        hit_browser = find_et(voyager_h5, config, 
                        gulp_size=2**18,  # Note: intentionally smaller than 2**20 to test slice offset
                        filename_out='./test_voyager_hits.csv',
                        log_output=True,
                        log_config=True
                        )
                        
        dframe = hit_browser.hit_table
        print(dframe[['f_start', 'snr', 'channel_idx', 'gulp_channel_idx', 'drift_rate', 'extent_lower', 'extent_upper']])

        # Third time -- add in poly fit
        config['preprocess']['poly_fit'] = 5
        hit_browser = find_et(voyager_h5, config, 
                        gulp_size=2**18,  # Note: intentionally smaller than 2**20 to test slice offset
                        filename_out='./test_voyager_hits.csv',
                        log_output=True,
                        log_config=True
                        )
                        
        dframe = hit_browser.hit_table
        print(dframe.dtypes)

        db = HitDatabase('test_voyager_hits.hitdb', mode='w')
        hit_browser.to_db(db, 'voyager')

    finally:
        for file_ext in ('.log', '.csv', '.yaml'):
            cleanup=True
            if cleanup:
                if os.path.exists('test_voyager_hits' + file_ext):
                    os.remove('test_voyager_hits' + file_ext)

if __name__ == "__main__":
    dframe = test_with_voyager()
