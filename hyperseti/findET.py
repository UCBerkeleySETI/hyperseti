""" Command-line utility for hyperseti find_et """

import time
from datetime import timedelta
from argparse import ArgumentParser
import logbook
from .log import update_levels
from hyperseti import find_et
from hyperseti.version import HYPERSETI_VERSION


def cmd_tool(args=None):
    r"""Coomand line parser"""
    parser = ArgumentParser(description=f"Hyperseti findET command-line utility, version {HYPERSETI_VERSION}.")
    parser.add_argument("input_path", type=str, help="Path of input file.")
    parser.add_argument("--output_csv_path", "-o", type=str, default="./hits.csv",
                        help="Output path of CSV file.  Default: ./hits.csv.")
    parser.add_argument("--gulp_exponent", "-z", type=int, default=18,
                        help="The power of 2 used in calculating the gulp size = number of fine channels to process in one gulp. Defaults to 18 (i.e. 2^18=262144).")
    parser.add_argument("--max_drift_rate", "-M", type=float, default=4.0,
                        help="Maximum doppler drift in Hz/s for searching.  Default: 4.0.")
    parser.add_argument("--min_drift_rate", "-m", type=float, default=0.001,
                        help="Minimum doppler drift in Hz/s for searching.  Default: 0.001.")
    parser.add_argument("--snr_threshold", "-s", type=float, default=30.0,
                        help="Minimum SNR value for searching.  Default: 30.0.")
    parser.add_argument("--boxcar_exponent", "-b", type=int, default=1,
                        help="The power of 2 used in calculating the number of boxcar trials to do. Default: 1.")
    parser.add_argument("--kernel", "-k", type=str, default="ddsk", choices=["dedoppler", "kurtosis", "ddsk"],
                        help="Kernel to be used by the dedoppler module.  Default: ddsk.")
    parser.add_argument("--gpu_id", "-g", type=int, default=0,
                        help="ID of GPU device.  Default: 0.")
    parser.add_argument("--group_level", "-l", type=str, default="info", choices=["debug", "info", "warning"],
                        help="Level for all functions that are not being debugged.  Default: info.")
    parser.add_argument("--debug_list", "-d", nargs="+", default=[],
                        help="List of logger names to use level=logbook.DEBUG.  Default: nil.")
    parser.add_argument("--noskflag", "-F",  default=False, action="store_true",
                        help="Do NOT apply spectral kurtosis flagging when normalizing data.")
    parser.add_argument("--nonormalize", "-N", default=False, action="store_true",
                        help="Do NOT normalize input data.")
    parser.add_argument("--nosmearcorr", "-S", default=False, action="store_true",
                        help="Do NOT apply doppler smearing correction.")
    parser.add_argument("--nomergeboxcar", "-X", default=False, action="store_true",
                        help="Do NOT merge boxcar trials.")
    parser.add_argument("--logfile", "-L", type=str, default="hits.log",
                        help="Name of logfile to write to")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    if args.group_level == "debug":
        int_level = logbook.DEBUG
    else:
        if args.group_level == "info":
            int_level = logbook.INFO
        else:
            int_level = logbook.WARNING


    pipeline_config = {
        'preprocess': {
            'sk_flag': not args.noskflag,
            'normalize': not args.nonormalize,
        },
        'dedoppler': {
            'kernel': args.kernel,
            'max_dd': args.max_drift_rate,
            'min_dd': args.min_drift_rate,
            'apply_smearing_corr': not args.nosmearcorr
        },
        'hitsearch': {
            'threshold': args.snr_threshold
        },
        'pipeline': {
            'n_boxcar': args.boxcar_exponent,
            'merge_boxcar_trials': not args.nomergeboxcar
        }
    }

    # Set the logbook levels for all of the functions.
    update_levels(int_level, args.debug_list)
    flogger =  logbook.FileHandler(args.logfile, mode='w', level=logbook.INFO)
    
    with flogger.applicationbound():
    
        # Find E.T.
        time1 = time.time()
        dframe = find_et(args.input_path, 
                        pipeline_config,
                        gulp_size=2**args.gulp_exponent,
                        gpu_id=args.gpu_id)
        time2 = time.time()

        time_delta = timedelta(seconds=(time2 - time1))
        print("findET: Output dataframe:\n", dframe)


if __name__ == "__main__":
    cmd_tool()
