"""
Try to get results that are similar to those of turbo_seti with our standard Voyager 1 .h5 file.

# ----------------------------------------------------
# File ID: xx.h5 
# ----------------------------------------------------
# Source:Voyager1
# MJD: 57650.782094907408       RA: 17h10m03.984s       DEC: 12d10m58.8s
# DELTAT:  18.253611    DELTAF(Hz):  -2.793968  max_drift_rate:   4.000000      obs_length: 292.057776
# --------------------------
# Top_Hit_#     Drift_Rate      SNR     Uncorrected_Frequency   Corrected_Frequency     Index   freq_start      freq_end        SEFD    SEFD_freq       Coarse_Channel_Number   Full_number_of_hits 
# --------------------------
001      -0.392226      156.379913         8419.319368     8419.319368  64651      8419.321003     8419.317740  0.0           0.000000  0       77634
002      -0.373093      1258.281250        8419.297028     8419.297028  72647      8419.298662     8419.295399  0.0           0.000000  0       77634
003      -0.392226      159.497269         8419.274374     8419.274374  80755      8419.276009     8419.272745  0.0           0.000000  0       77634

"""
import pandas as pd
import setigen as stg
import astropy.units as u
from hyperseti.blanking import blank_hits
from hyperseti.io import from_setigen
from hyperseti.pipeline import GulpPipeline

def test_blank_hits():
    metadata = {'fch1': 6095.214842353016*u.MHz, 
                'dt': 18.25361108*u.s, 
                'df': 2.7939677238464355*u.Hz}

    frame = stg.Frame(fchans=2**12*u.pixel,
                    tchans=32*u.pixel,
                    df=metadata['df'],
                    dt=metadata['dt'],
                    fch1=metadata['fch1'])

    start_idxs = 500, 700, 2048, 3000
    test_tones = [
    {'f_start': frame.get_frequency(index=start_idxs[0]), 'drift_rate': 0.70*u.Hz/u.s, 'snr': 100, 'width': 20*u.Hz},
    {'f_start': frame.get_frequency(index=start_idxs[1]), 'drift_rate': -0.55*u.Hz/u.s, 'snr': 100, 'width': 20*u.Hz},
    {'f_start': frame.get_frequency(index=start_idxs[2]), 'drift_rate': 0.00*u.Hz/u.s, 'snr': 40, 'width': 6*u.Hz},
    {'f_start': frame.get_frequency(index=start_idxs[3]), 'drift_rate': 0.07*u.Hz/u.s, 'snr': 50, 'width': 3*u.Hz}
    ]

    noise = frame.add_noise(x_mean=10, x_std=5, noise_type='chi2')

    for tone in test_tones:
        signal = frame.add_signal(stg.constant_path(f_start=tone['f_start'],
                                                drift_rate=tone['drift_rate']),
                            stg.constant_t_profile(level=frame.get_intensity(snr=tone['snr'])),
                            stg.gaussian_f_profile(width=tone['width']),
                            stg.constant_bp_profile(level=1))

    d = from_setigen(frame)
    d = d.isel({'frequency': slice(0, 2**12)}, space='gpu')

    config = {
        'preprocess': {
            'sk_flag': True,
            'normalize': True,
        },
        'sk_flag': {
            'n_sigma': 1,
        },
        'dedoppler': {
            'kernel': 'ddsk',
            'max_dd': 4.0,
            'min_dd': None,
            'apply_smearing_corr': False
        },
        'hitsearch': {
            'threshold': 3,
            'min_fdistance': 100
        },
        'pipeline': {
            'n_boxcar': 4,
            'merge_boxcar_trials': True
        }
    }
    pipeline = GulpPipeline(d, config)
    df = pipeline.run()
    db = blank_hits(d, df)

    for idx in start_idxs:
        assert db.data[0,0, idx] == 0
    print("Hits blanked!")

if __name__ == "__main__":
    test_blank_hits()
