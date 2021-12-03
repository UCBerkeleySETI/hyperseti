import logbook
from argparse import ArgumentParser

from hyperseti import find_et

def cmd_tool(args=None):
    r"""Coomand line parser"""
    parser = ArgumentParser(description="Make waterfall plots from a single file.")
    parser.add_argument("input_path", type=str, help="Path of input file.")
    parser.add_argument("--output_csv_path", "-o", type=str, default="./hits.csv",
                        help="Output path of CSV file.  Default: ./hits.csv.")
    parser.add_argument("--gulp_size", "-z", type=int, default=2**19,
                        help="Number of channels to process at once (e.g. Number of fine channels in a coarse channel).  Default: 2**19.")
    parser.add_argument("--max_drift_rate", "-M", type=float, default=4.0,
                        help="Maximum doppler drift in Hz/s for searching.  Default: 4.0.")
    parser.add_argument("--min_drift_rate", "-m", type=float, default=0.001,
                        help="Minimum doppler drift in Hz/s for searching.  Default: 0.001.")
    parser.add_argument("--snr_threshold", "-s", type=float, default=30.0,
                        help="Minimum SNR value for searching.  Default: 30.0.")
    parser.add_argument("--num_boxcars", "-b", type=int, default=6,
                        help="Number of boxcar trials to do, width 2^N e.g. trials=(1,2,4,8,16).  Default: 6.")
    parser.add_argument("--gpu_id", "-g", type=int, default=0,
                        help="ID of GPU device.  Default: 0.")
    parser.add_argument("--log_level", "-l", type=str, default="info", choices=["debug", "info", "warning"],
                        help="Logbook level.  Default: info.")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    if args.log_level == "debug":
        int_level = logbook.DEBUG
    else:
        if args.log_level == "info":
            int_level = logbook.INFO
        else:
            int_level = logbook.WARNING

    dframe = find_et(args.input_path, 
                    filename_out=args.output_csv_path, 
                    gulp_size=args.gulp_size, 
                    max_dd=args.max_drift_rate, 
                    min_dd=args.min_drift_rate,
                    n_boxcar=args.num_boxcars,
                    threshold=args.snr_threshold,
                    gpu_id=args.gpu_id,
                    log_level=int_level)

    print("\nfindET: Output dataframe:\n", dframe)

if __name__ == "__main__":
    cmd_tool()