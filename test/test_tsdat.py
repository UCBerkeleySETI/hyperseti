import os
from os import path
import shutil
import time
from blimpy.utils import change_the_ext
from turbo_seti import run_pipelines
from hyperseti import find_et
from hyperseti.tsdat import cmd_tool as gendat
from .file_defs import voyager_h5

# File path definitions.
HERE = path.dirname(path.abspath(__file__))
PATH_TEST_DIR = path.join(HERE, "test_tsdat")
PATH_HITS_CSV = PATH_TEST_DIR + "/hits.csv"
PATH_DAT = change_the_ext(voyager_h5, "h5", "dat")


# hyperseti parameters:
GULP_SIZE = 107375
N_BOXCAR = 6
KERNEL = "dedoppler"
MAX_DRIFT_RATE = 4.0
MIN_DRIFT_RATE = 0.00001
SNR_THRESHOLD = 25.0
GPU_ID = 3


def test_tsdat():

    # Set up directory with symlink to Voyager 1 .h5 file.
    shutil.rmtree(PATH_TEST_DIR, ignore_errors=True)
    os.mkdir(PATH_TEST_DIR)
    voyager_filename = voyager_h5.split("/")[-1]
    PATH_H5 = PATH_TEST_DIR + "/" + voyager_filename
    os.symlink(voyager_h5, PATH_H5, target_is_directory = False)

    # Run findET
    t1 = time.time()
    config = {
        'preprocess': {
            'sk_flag': True,
            'normalize': True,
        },
        'dedoppler': {
            'boxcar_mode': 'sum',
            'kernel': 'ddsk',
            'max_dd': 4.0,
            'min_dd': None,
            'apply_smearing_corr': True,
            'beam_id': 0
        },
        'hitsearch': {
            'threshold': 20,
            'min_fdistance': 100
        },
        'pipeline': {
            'n_boxcar': 1,
            'merge_boxcar_trials': True
        }
    }
    
    dframe = find_et(voyager_h5, config, filename_out=PATH_HITS_CSV, gulp_size=2**18)

    et = time.time() - t1
    print(f"findET E.T. is {et:.1f} seconds")

    # Create a .dat file from the .h5 file and the .csv file.
    parms = ["-M", str(MAX_DRIFT_RATE), "-l", "16", "-o", PATH_TEST_DIR, PATH_HITS_CSV, PATH_H5]
    gendat(parms)

    # Create turbo_seti plots.
    parms = [PATH_TEST_DIR, "-o", PATH_TEST_DIR]
    run_pipelines.main(parms)
