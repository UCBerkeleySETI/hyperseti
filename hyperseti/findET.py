""" Command-line utility for hyperseti find_et """

import sys
import time
from datetime import timedelta
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import logbook
from .log import update_levels
from hyperseti import find_et
from hyperseti.version import HYPERSETI_VERSION


HELP_EPILOGUE = \
"""
Parameters expressed as exponents
---------------------------------
The --gulp_exponent (-z) parameter value is an integer N such that the actual gulp size used is 2^N.
For example, if the -z value is 4, then the gulp size used by hyperseti is 16.
The --boxcar_exponent (-b) parameter has a similar function.

Command-line note about --debug_list (-d)
-----------------------------------------
This parameter is a variable-length list. As such, it should be specified at the end of the command line (E.g. bash).
Specified at the end would avoid confusion on the part of argument parsing (Python ArgParser).
E.g. findET Voyager1.single_coarse.fine_res.h5 -M 4 -s 25 -g 3 -d hyperseti.hits hyperseti.normalize
"""

brief_desc = f"Hyperseti findET command-line utility, version {HYPERSETI_VERSION}."


def cmd_tool(args=None):
    r"""Coomand line parser"""
    parser = ArgumentParser(description=brief_desc,
                formatter_class=RawDescriptionHelpFormatter,
                epilog=HELP_EPILOGUE)
    parser.add_argument("input_path", type=str, help="Path of input file.", nargs="?", default=None)
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
    parser.add_argument("--poly_fit", "-t", type=int, default=0,
                        help="Degree of least-squares fit polynomial used in hyperseti/normalize.  Default: 0 (unused).")
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
    parser.add_argument("--version", "-v", default=False, action="store_true",
                        help="Display version ID and exit to O/S.")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    if args.version:
        print(brief_desc)
        sys.exit(0)

    if args.input_path is None:
        print("*** An input file path must be supplied !")
        sys.exit(1)

    if args.group_level == "debug":
        int_level = logbook.DEBUG
    else:
        if args.group_level == "info":
            int_level = logbook.INFO
        else:
            int_level = logbook.WARNING

    # Create pipeline configuration object for find_et.
    pipeline_config = {
        'preprocess': {
            'sk_flag': not args.noskflag,
            'normalize': not args.nonormalize,
            'poly_fit': args.poly_fit
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
                        filename_out=args.output_csv_path,
                        gulp_size=2**args.gulp_exponent,
                        gpu_id=args.gpu_id)
        time2 = time.time()

        time_delta = timedelta(seconds=(time2 - time1))
        print("findET: Output dataframe:\n", dframe)


if __name__ == "__main__":
    cmd_tool()
