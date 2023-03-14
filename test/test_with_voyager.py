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

"""

import os
import logbook
import numpy as np
from hyperseti import find_et
from hyperseti.log import update_levels, get_logger
from hyperseti.utils import time_logger

try:
    from .file_defs import synthetic_fil, test_fig_dir, voyager_h5
except:
    from file_defs import synthetic_fil, test_fig_dir, voyager_h5

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
        time_logger.level = logbook.INFO

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
                'threshold': 20,
                'min_fdistance': None
            },
            'pipeline': {
                'n_boxcar': 10,
                'merge_boxcar_trials': True
            }
        }

        dframe = find_et(voyager_h5, config, 
                        gulp_size=2**18,  # Note: intentionally smaller than 2**20 to test slice offset
                        filename_out='./test_voyager_hits.csv',
                        log_output=True,
                        log_config=True
                        )
    
        # dframe column names: drift_rate  f_start  snr  driftrate_idx  channel_idx  boxcar_size  beam_idx  n_integration
        print("Returned dataframe:\n", dframe)
        list_drate = dframe["drift_rate"].tolist()

        assert os.path.exists('test_voyager_hits.csv')
        assert os.path.exists('test_voyager_hits.yaml')
        assert os.path.exists('test_voyager_hits.log')

        # This is a quick test to check if smaller gulps are taking the channel offset into account
        assert np.alltrue(dframe['channel_idx'] > 739000)

        for drate in list_drate:
            print("Observed drift rate = {}, should be negative.".format(drate))
            assert drate <= 0.0
        return dframe

    finally:
        for file_ext in ('.log', '.csv', '.yaml'):
            if os.path.exists('test_voyager_hits' + file_ext):
                os.remove('test_voyager_hits' + file_ext)

if __name__ == "__main__":
    dframe = test_with_voyager()
