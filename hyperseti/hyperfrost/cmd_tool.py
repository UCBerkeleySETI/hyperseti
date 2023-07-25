""" Command-line utility for hyperseti find_et """

import glob, os, pprint
from datetime import datetime

import pandas as pd
import numpy as np
import cupy as cp
import os
from pprint import pprint
from copy import deepcopy

from hyperseti.io import load_config
from hyperseti.hyperfrost.data_source import DataArrayBlock
from hyperseti.hyperfrost.hyperfrost_block import hyperfrost_pipeline

import bifrost as bf
import bifrost.pipeline as bfp

from astropy.time import Time
import sys
import time
from datetime import timedelta
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import logbook
from hyperseti.version import HYPERSETI_VERSION


brief_desc = f"Hyperfrost command-line utility, bifrost version {bf.__version__}, hyperseti version {HYPERSETI_VERSION}."

def cmd_tool(args=None):
    r"""Command line parser"""
    parser = ArgumentParser(description=brief_desc,
                formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("config", type=str, help="Path of pipeline configurational YAML file.", nargs='?', default=None)
    parser.add_argument("-v", "--version", default=False, action="store_true",
                        help="Display version ID and exit to O/S.")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    if args.version:
        print(brief_desc)
        sys.exit(0)

    config = load_config(args.config)
    h = config['hyperfrost']

    dt0 = datetime.utcnow()
    ts0  = Time(dt0).isot

    pprint.pprint(config)
    filelist = sorted(glob.glob(os.path.join(h['input_path'], h['file_ext'])))

    hit_db      = os.path.join(h['output_path'], f"h['db_prefix']_{ts0}.hitdb")            # Output HitDatabase name

    ##############
    ## BIFROST PIPELINE
    ##############

    n_workers = h.get('n_workers', 1)
    overlap   = h.get('gulp_overlap', 0)

    b_read      = DataArrayBlock(filelist, h['gulp_size'], axis='frequency', overlap=overlap)
    b_hyper     = hyperfrost_pipeline(b_read, n_workers, config, db=hit_db)

    pipeline = bfp.get_default_pipeline()
    print(pipeline.dot_graph())
    pipeline.run()

    # Get time taken
    dt1 = datetime.utcnow()
    ts1  = Time(dt1).isot
    print(f"Time taken: {dt1 - dt0}")

if __name__ == "__main__":
    cmd_tool()
