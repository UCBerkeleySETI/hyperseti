"""parkes_uwl.py: Run hyperseti on a UWL session at BLPD.

Runs a hit search on all blcXY subfolders within a session folder,
e.g. /datag/collate_mb/PKS_0620_2020-06-11T11:00
"""
import glob, os, pprint
from datetime import datetime
from hyperseti import find_et
from hyperseti.io import load_config
from hyperseti.io.hit_db import HitDatabase
from astropy.time import Time

fpath       = '/datag/collate_mb/PKS_0620_2020-06-11T11:00'
config_file = 'pks_for_joe.yaml'                    # Pipeline config file to load
hit_db      = '2024_07_18-pks_uwl.hitdb'            # Output HitDatabase name
hires_ext   = '*.0000.h5'                           # Extension to use to ID hi-freq res data
gpu_id      = 0                                     # GPU to attach to (blpc nodes can be busy!)
gulp_size   = 2**19                                 # Read about 1 MHz data (2^19 = 524288, chan_bw = 2 Hz)
n_overlap   = 1024                                  # Overlap read by 1024 channels

folders_to_skip = ('blc00', 'blc01', 'blc02', 'blc03')  # Skipping low-band 

# Setup
dt0 = datetime.utcnow()
ts  = Time(dt0).isot
errlog_fn = f'errors_{ts}.log'

# Load config and glob folder list
blc_folders = sorted(glob.glob(os.path.join(fpath, 'blc*')))
for dirname in folders_to_skip:
    dpath = f'/datag/collate_mb/PKS_0620_2020-06-11T11:00/{dirname}'
    if dpath in blc_folders:
        blc_folders.pop(blc_folders.index(dpath))

config = load_config(config_file)
pprint.pprint(config)

print("Subfolders:")
pprint.pprint(blc_folders)

# Create new hit database
db = HitDatabase(hit_db, mode='w')

# Loop through blcXX folders, and search for hires H5 files
for ff, folder in enumerate(blc_folders):
    filelist = sorted(glob.glob(os.path.join(folder, hires_ext)))
    for ii, filename in enumerate(filelist):
        print(f"(node {ff+1} / {len(blc_folders)}: file {ii + 1}/{len(filelist)}) Opening {filename}...")
        
        # A slightly more friendly observation ID, prepending blcXX to ensure ID is unique
        blc_id = os.path.basename(folder)
        obs_id = os.path.basename(filename)
        obs_id = obs_id.replace('guppi_', '').replace(hires_ext,'')
        obs_id = f'{blc_id}_{obs_id}'
	
	
        try:
            # Search for hits, file by file. Do not save to file here as we save to HitDatabase
            hit_browser = find_et(filename, config, 
                        gulp_size=gulp_size,
                        n_overlap=n_overlap,
			gpu_id=gpu_id,
                        filename_out=None, log_output=False, log_config=False
                        )

            # Save to the HitDatabase 
            print(f"Saving to obs_id: {obs_id}")
            hit_browser.to_db(db, obs_id=obs_id)
        except KeyboardInterrupt:
            import sys
            print("Keyboard interrupt, exiting...")
            sys.exit(0)
        except:
            with open(f'errors.log', 'a') as eh:
                now = datetime.utcnow()
                eh.write(f"[{now}] {filename}\n")
                import time
                time.sleep(0.25)
                raise
