from hyperseti.dedoppler import dedoppler
from hyperseti.hits import hitsearch, merge_hits
from hyperseti.io import from_fil, from_h5, from_setigen
from hyperseti.log import set_log_level
from hyperseti.pipeline import GulpPipeline
import cupy as cp

from astropy import units as u
import setigen as stg
import pylab as plt
import numpy as np

from hyperseti.plotting import imshow_dedopp, imshow_waterfall, overlay_hits

from hyperseti.test_data import voyager_h5, voyager_yaml, tmp_file, tmp_dir
import os

synthetic_fil = tmp_file('synthetic.fil')
test_fig_dir = tmp_dir('test_figs')

def test_hitsearch_plotting():
    """ Test hit search routine with multiple signals """

    set_log_level("info")
    metadata_in = {'frequency_start': 6095.214842353016*u.MHz,
            'time_step': 18.25361108*u.s,
            'frequency_step': 2.7939677238464355*u.Hz}

    frame = stg.Frame(fchans=2**16*u.pixel,
                      tchans=32*u.pixel,
                      df=metadata_in['frequency_step'],
                      dt=metadata_in['time_step'],
                      fch1=metadata_in['frequency_start'])

    frame.add_noise(x_mean=1, x_std=5, noise_type='chi2')

    test_tones = [
      {'f_start': frame.get_frequency(index=500), 'drift_rate': 0.50*u.Hz/u.s, 'snr': 1000, 'width': 20*u.Hz},
      {'f_start': frame.get_frequency(index=800), 'drift_rate': -0.40*u.Hz/u.s, 'snr': 500, 'width': 20*u.Hz},
      {'f_start': frame.get_frequency(index=2048), 'drift_rate': 0.00*u.Hz/u.s, 'snr': 100, 'width': 6*u.Hz},
      {'f_start': frame.get_frequency(index=3000), 'drift_rate': 0.07*u.Hz/u.s, 'snr': 200, 'width': 3*u.Hz}
    ]

    for tone in test_tones:
        frame.add_signal(stg.constant_path(f_start=tone['f_start'],
                                                drift_rate=tone['drift_rate']),
                              stg.constant_t_profile(level=frame.get_intensity(snr=tone['snr'])),
                              stg.gaussian_f_profile(width=tone['width']),
                              stg.constant_bp_profile(level=1))

    frame.save_fil(filename=synthetic_fil)
    darray = from_fil(synthetic_fil)
    darray.data = cp.asarray(darray.data)

    config = {
        'preprocess': {
            'sk_flag': False,
            'normalize': True,
        },
        'dedoppler': {
            'kernel': 'ddsk',
            'max_dd': 1.0,
            'min_dd': None,
            'apply_smearing_corr': True,
        },
        'hitsearch': {
            'threshold': 5,
            'min_fdistance': 100
        },
        'pipeline': {
            'n_boxcar': 5,
            'n_blank': 2,
            'merge_boxcar_trials': True
        }
    }
    pipeline = GulpPipeline(darray, config)
    hits = pipeline.run()
    
    print(hits.sort_values('snr', ascending=False))

    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    imshow_waterfall(darray, xaxis='channel', yaxis='timestep')

    plt.subplot(1,2,2)
    dedopp = dedoppler(darray, max_dd=1.0, min_dd=None)

    imshow_dedopp(dedopp, xaxis='channel', yaxis='driftrate')
    overlay_hits(hits, 'channel', 'driftrate')

    plt.savefig(os.path.join(test_fig_dir, 'test_hitsearch_multi.png'))
    plt.show()
    plt.clf()

    imshow_dedopp(dedopp, xaxis='frequency', yaxis='driftidx')
    overlay_hits(hits, 'frequency', 'driftidx')

    plt.savefig(os.path.join(test_fig_dir, 'test_hitsearch_multi_v2.png'))
    plt.show()

if __name__ == "__main__":
    test_hitsearch_plotting()