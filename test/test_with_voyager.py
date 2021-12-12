import os
import logbook
from hyperseti import find_et
from hyperseti.log import update_levels

from .file_defs import synthetic_fil, test_fig_dir, voyager_h5

# Other parameters:
GULP_SIZE = 1048576
GULP_SIZE = 107375
MAX_DRIFT_RATE = 4.0
MIN_DRIFT_RATE = 0.01
SNR_THRESHOLD = 30.0
N_BOXCAR = 6
KERNEL = "dedoppler"
GPU_ID = 0


def test_with_voyager():
    print("hyperseti find_et from file {} .....".format(voyager_h5))
    update_levels(logbook.INFO, [])
    dframe = find_et(voyager_h5, 
                    filename_out='./hyperseti_hits.csv', 
                    gulp_size=GULP_SIZE, 
                    max_dd=MAX_DRIFT_RATE, 
                    min_dd=MIN_DRIFT_RATE,
                    n_boxcar=N_BOXCAR,
                    kernel=KERNEL,
                    gpu_id=GPU_ID,
                    threshold=SNR_THRESHOLD)
    print("Returned dataframe:", dframe)
    list_drate = dframe["drift_rate"].tolist()
    for drate in list_drate:
        print("Observed drift rate = {}, should be negative.".format(drate))
        assert drate < 0.0
