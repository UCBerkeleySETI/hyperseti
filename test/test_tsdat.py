import os
from os import path
import shutil
import time
from blimpy.utils import change_the_ext
from turbo_seti import run_pipelines
from hyperseti import find_et
from hyperseti.tsdat import cmd_tool as gendat
from hyperseti.test_data import voyager_h5, tmp_dir
import pytest

# File path definitions.
PATH_TEST_DIR = tmp_dir("test_tsdat")
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

v_sym = os.path.join(PATH_TEST_DIR, 'voyager.h5')
if not os.path.exists(v_sym):
    os.symlink(voyager_h5, v_sym, target_is_directory=False)


def test_tsdat():


    # Run findET
    t1 = time.time()
    config = {
        'preprocess': {
            'sk_flag': True,
            'normalize': True,
        },
        'dedoppler': {
            'kernel': 'ddsk',
            'max_dd': 4.0,
            'min_dd': None,
            'apply_smearing_corr': True,
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
    parms = ["-M", str(MAX_DRIFT_RATE), "-l", "16", "-o", PATH_TEST_DIR, PATH_HITS_CSV, v_sym]
    gendat(parms)

    # Create turbo_seti plots.
    parms = [PATH_TEST_DIR, "-o", PATH_TEST_DIR]
    run_pipelines.main(parms)


def test_tsdat_errors():
    import mock
    with pytest.raises(SystemExit):
        parms = ["-M", str(MAX_DRIFT_RATE), "-l", "16", "-o", PATH_TEST_DIR, 'missing_file.csv', v_sym]
        gendat(parms)

    with pytest.raises(SystemExit):
        parms = ["-M", str(MAX_DRIFT_RATE), "-l", "16", "-o", PATH_TEST_DIR, PATH_HITS_CSV, 'missing_file.h5']
        gendat(parms)

if __name__ == "__main__":
    test_tsdat()
    test_tsdat_errors()