import os
from shutil import rmtree

import h5py
import hdf5plugin

from file_defs import voyager_fil, voyager_h5, voyager_h5_flipped

DATADIR = './test_data/'
FIGDIR = './test_figs/'


def download_test_data():
    """ Download Voyager test data """
    os.system('curl --url "http://blpd0.ssl.berkeley.edu/Voyager_data/Voyager1.single_coarse.fine_res.h5"  -o ' + DATADIR + 'Voyager1.single_coarse.fine_res.h5')
    os.system('curl --url "http://blpd0.ssl.berkeley.edu/Voyager_data/Voyager1.single_coarse.fine_res.fil"  -o ' + DATADIR + 'Voyager1.single_coarse.fine_res.fil')


def flip_data():
    """ Flip Voyager data along frequency axis.

    The flipped file is used to check logic works when frequency is inverted.
    """
    print("Generating frequency flipped version of Voyager data...")
    os.system('cp %s %s' % (voyager_h5, voyager_h5_flipped))
    with h5py.File(voyager_h5_flipped, 'r+') as h:
        foff_orig = h['data'].attrs['foff']
        fch1_orig = h['data'].attrs['fch1']
        nchans    = h['data'].attrs['nchans']
        fchN      = fch1_orig + (foff_orig * nchans)
        h['data'].attrs['foff'] = foff_orig * -1
        h['data'].attrs['fch1'] = fchN
        h['data'].attrs['source_name'] = 'Voyager1Flipped'

        for ii in range(h['data'].shape[0]):
            print('\tFlipping %i/%i' % (ii+1, h['data'].shape[0]))
            h['data'][ii, 0, :] = h['data'][ii, 0][::-1]
    print("Done.")


if __name__ == "__main__":

    if not os.path.exists(DATADIR):
        print(f"Creating directory {DATADIR}")
        os.mkdir(DATADIR)

    if not os.path.exists(FIGDIR):
        print(f"Creating directory {FIGDIR}")
        os.mkdir(FIGDIR)
        
    download_test_data()
    flip_data()
