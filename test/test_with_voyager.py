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
from hyperseti import find_et
from hyperseti.log import update_levels

from hyperseti.io.filterbank import from_fil
from hyperseti.io.hdf5 import from_h5
from .file_defs import voyager_h5

# Other parameters:
GULP_SIZE = 1048576
MAX_DRIFT_RATE = 4.0
MIN_DRIFT_RATE = 0.0
SNR_THRESHOLD = 30.0
N_BOXCAR = 2
KERNEL = "dedoppler"
GPU_ID = 0


def test_with_voyager():
    print("hyperseti find_et from file {} .....".format(voyager_h5))
    dframe = find_et(voyager_h5, 
                    filename_out='./hyperseti_hits.csv', 
                    gulp_size=GULP_SIZE, 
                    max_dd=MAX_DRIFT_RATE, 
                    min_dd=MIN_DRIFT_RATE,
                    n_boxcar=N_BOXCAR,
                    kernel=KERNEL,
                    gpu_id=GPU_ID,
                    threshold=SNR_THRESHOLD)
    
    # dframe column names: drift_rate  f_start  snr  driftrate_idx  channel_idx  boxcar_size  beam_idx  n_integration
    print("Returned dataframe:\n", dframe)
    list_drate = dframe["drift_rate"].tolist()
    for drate in list_drate:
        print("Observed drift rate = {}, should be negative.".format(drate))
        assert drate < 0.0
    return dframe

if __name__ == "__main__":
    dframe = test_with_voyager()
