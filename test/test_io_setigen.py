from hyperseti.io import from_setigen, load_data
import setigen as stg
from astropy import units as u

def test_setigen():
    """ Generate a setigen frame and convert it into a DataArray"""
    metadata = {'fch1': 6095.214842353016*u.MHz, 
                'dt': 18.25361108*u.s, 
                'df': 2.7939677238464355*u.Hz}

    frame = stg.Frame(fchans=2**12*u.pixel,
                    tchans=128*u.pixel,
                    df=metadata['df'],
                    dt=metadata['dt'],
                    fch1=metadata['fch1'])

    test_tones = [
    {'f_start': frame.get_frequency(index=500), 'drift_rate': 0.01*u.Hz/u.s, 'snr': 100, 'width': 20*u.Hz},
    {'f_start': frame.get_frequency(index=800), 'drift_rate': -0.10*u.Hz/u.s, 'snr': 100, 'width': 20*u.Hz},
    {'f_start': frame.get_frequency(index=2048), 'drift_rate': 0.00*u.Hz/u.s, 'snr': 20, 'width': 6*u.Hz},
    {'f_start': frame.get_frequency(index=3000), 'drift_rate': 0.07*u.Hz/u.s, 'snr': 50, 'width': 3*u.Hz}
    ]

    noise = frame.add_noise(x_mean=100, x_std=5, noise_type='gaussian')

    for tone in test_tones:
        signal = frame.add_signal(stg.constant_path(f_start=tone['f_start'],
                                                drift_rate=tone['drift_rate']),
                            stg.constant_t_profile(level=frame.get_intensity(snr=tone['snr'])),
                            stg.gaussian_f_profile(width=tone['width']),
                            stg.constant_bp_profile(level=1))

    d = from_setigen(frame)

    load_data(frame)

def test_load_data():
    with pytest.raises(RuntimeError):
        a = np.array([1,2,3])
        load_data(a)
    
if __name__ == "__main__":
    test_setigen()
    test_load_data()