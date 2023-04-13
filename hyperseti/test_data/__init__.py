from astropy.utils.data import conf, download_file
import os
import h5py
import hdf5plugin

# Zenodo can be slow to respond, so set timeout to 30 seconds
conf.remote_timeout = 30

HERE = os.path.dirname(os.path.abspath(__file__))

# Cache dir has general form of /home/dancpr/.hyperseti/cache/download/
voyager_h5 = download_file("http://blpd0.ssl.berkeley.edu/Voyager_data/Voyager1.single_coarse.fine_res.h5",
                           pkgname='hyperseti', cache=True, show_progress=True)

voyager_fil = download_file("http://blpd0.ssl.berkeley.edu/Voyager_data/Voyager1.single_coarse.fine_res.fil",
                            pkgname='hyperseti', cache=True, show_progress=True)                           

CACHE_DIR = os.path.join(os.path.dirname(voyager_h5), '../../../')
if not os.path.exists(os.path.join(CACHE_DIR, 'tmp')):
    os.mkdir(os.path.join(CACHE_DIR, 'tmp'))

def flip_data(voyager_h5: str): #pragma: no cover
    """ Flip Voyager data along frequency axis.

    The flipped file is used to check logic works when frequency is inverted.
    """
    voyager_h5_flipped = os.path.join(CACHE_DIR, 'Voyager1.single_coarse.fine_res.flipped.h5')
    if not os.path.exists(voyager_h5_flipped):
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
    return voyager_h5_flipped

voyager_h5_flipped = flip_data(voyager_h5)
voyager_csv  = os.path.join(HERE, 'Voyager1.single_coarse.fine_res.csv')
voyager_yaml = os.path.join(HERE, 'Voyager1.single_coarse.fine_res.yaml')

def tmp_file(filename: str) -> str:
    """ Create a temporary file in ~/.hyperseti"""
    if not os.path.exists(os.path.join(CACHE_DIR, 'tmp')):
        os.mkdir(os.path.join(CACHE_DIR, 'tmp'))
    tmp_filename = os.path.join(CACHE_DIR, 'tmp', filename)
    return tmp_filename

def tmp_dir(dirname: str) -> str:
    """ Create a temporary directory in ~/.hyperseti"""
    if not os.path.exists(os.path.join(CACHE_DIR, 'tmp', dirname)):
        os.mkdir(os.path.join(CACHE_DIR, 'tmp', dirname))
    return os.path.join(CACHE_DIR, 'tmp', dirname)
