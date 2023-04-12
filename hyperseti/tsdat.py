""" tsdat - Create a turbo_seti .dat file """

import sys
import os
from os.path import exists
from argparse import ArgumentParser
import pandas as pd
from blimpy import Waterfall

DFCOLS = ["drift_rate", "f_start", "snr", "channel_idx"]


def write_header_lines(fd, h5_file, header, max_drift_rate, obs_length):
    r"""
    Write formatted header lines.

    Parameters
    ----------
    fd : File Descriptor Object
        File writer handle.
    header : dict
        source_name : str
            Name of the source being observed.
        tstart (MJD) : float
            Time stamp (MJD) of first sample.
        src_raj : str
            Right ascension (J2000) of source (hhmmss.s).
        src_dej : str
            Declination (J2000) of source (ddmmss.s).
        tsamp : float (> 0.0)
            Time interval between samples (s).
        foff : float (signed)
            Fine channel bandwidth (MHz).
    max_drift_rate : float
        Maximum drift rate (Hz/s) for the entire observation.
    obs_len : float
        Total observation time (seconds).
    """
    filename = h5_file.split("/")[-1]
    fd.write("# -------------------------- o --------------------------\n")
    fd.write(f"# File ID: {filename} \n")
    fd.write("# -------------------------- o --------------------------\n")

    info_str_1 = f"# Source:{header['source_name']}\n"
    info_str_2 = f"# MJD: {header['tstart']:18.12f}\tRA: {str(header['src_raj'])}\tDEC: {str(header['src_dej'])}\n"
    info_str_3a = f"# DELTAT: {header['tsamp']:10.6f}\tDELTAF(Hz): {header['foff']:10.6f}"
    info_str_3b = f"\tmax_drift_rate: {max_drift_rate:10.6f}\tobs_length: {obs_length:10.6f}\n"

    fd.write(info_str_1)
    fd.write(info_str_2)
    fd.write(info_str_3a + info_str_3b)
    fd.write("# --------------------------\n")
    info_str = "# Top_Hit_# \t"
    info_str += "Drift_Rate \t"
    info_str += "SNR \t"
    info_str += "Uncorrected_Frequency \t"
    info_str += "Corrected_Frequency \t"
    info_str += "Index \t"
    info_str += "freq_start \t"
    info_str += "freq_end \t"
    info_str += "SEFD \t"
    info_str += "SEFD_freq \t"
    info_str += "Coarse_Channel_Number \t"
    info_str += "Full_number_of_hits \t"
    info_str += "\n"
    fd.write(info_str)
    fd.write("# --------------------------\n")


def write_tophit_line(fd, tophit_count, channel_idx, hit_freq, drift_rate, snr):
    r"""
    This function looks into the top hit in a region, basically finds the local maximum and saves that.

    Parameters
    ----------
    fd : File Descriptor Object
        File writer handle.
    tophits_count : int
        Current count of the number of top hits.
    channel_idx : int
    fftlen : int
      Length of the fast fourier transform matrix.
    drift_rate : float
    snr : float

    Returns
    -------
    Nothing

    """

    # TODO freq_start and freq_end:
         # Are they used in turbo_seti?
         # If so, an estimate is needed.
    uncorr_freq = hit_freq
    freq_end = freq_start = uncorr_freq

    # As with turbo_seti, there is no frequency correction.
    corr_freq = uncorr_freq

    info_str  = f"{tophit_count:06d}"   # Top Hit number
    info_str += f"\t{drift_rate:10.6f}"   # Drift Rate
    info_str += f"\t{snr:10.6f}"          # SNR
    info_str += f"\t{uncorr_freq:14.6f}"  # Uncorrected Frequency
    info_str += f"\t{corr_freq:14.6f}"    # Corrected Frequency
    info_str += f"\t{channel_idx:n}"    # Channel Index
    info_str += f"\t{freq_start:14.6f}"   # turbo_seti find_doppler freq_start
    info_str += f"\t{freq_end:14.6f}"     # turbo_seti find_doppler freq_end
    info_str += "\t0.0\t0.000000\t0\t0\n"    # turbo_seti SEFD, SEFD_mid_freq,
                                        #    coarse channel number,
                                        #    total number of candidate hits
    fd.write(info_str)


def cmd_tool(args=None):
    r"""Coomand line parser"""
    parser = ArgumentParser(description="Create a turbo_seti .dat file from the hyperseti output CSV and the HDF5 file.")
    parser.add_argument("csv_file", type=str, default=None, help="Input CSV file from hyperseti")
    parser.add_argument("h5_file", type=str, default=None, help="Input HDF5 file")
    parser.add_argument("--max_drift_rate", "-M", type=float, required=True,
                        help="Maximum drift rate (Hz/s).  Required.")
    parser.add_argument("--obs_len", "-l", type=float, required=True,
                        help="Observation length (s).  Required.")
    parser.add_argument("--outdir", "-o", type=str, default="./", help="Output DAT directry.  Default: ./")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    path_csv_file = os.path.abspath(args.csv_file)
    if not exists(path_csv_file):
        print(f"*** CSV file {args.csv_file} does not exist !")
        os.system("tsdat -h")
        sys.exit(1)

    path_h5_file = os.path.abspath(args.h5_file)
    if not exists(path_h5_file):
        print(f"*** HDF5 file {args.h5_file} does not exist !")
        os.system("tsdat -h")
        sys.exit(1)

    path_dat_dir = os.path.abspath(args.outdir)
    h5_filename = path_h5_file.split("/")[-1]
    dat_filename = h5_filename.replace(".h5", ".dat")
    path_dat_file = path_dat_dir + "/" + dat_filename

    with open(path_dat_file, mode="w", encoding="utf-8") as fd:
        tophit_count = 0
        h5 = Waterfall(path_h5_file, load_data=False)
        df = pd.read_csv(path_csv_file, encoding="utf-8", engine="python")
        df = df[DFCOLS]
        nrows = len(df)
        write_header_lines(fd, path_h5_file, h5.header, args.max_drift_rate, args.obs_len)
        if nrows < 1:
            print("*** Empty .dat file written !")
            sys.exit(0)
        for row in df.itertuples():
            tophit_count += 1
            drift_rate = float(row[1])
            hit_freq = float(row[2])
            snr = float(row[3])
            channel_idx = int(row[4])
            write_tophit_line(fd, tophit_count, channel_idx, hit_freq, drift_rate, snr)
