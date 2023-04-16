import cupy as cp
import numpy as np
import setigen as stg

from astropy import units as u
from hyperseti.io import from_setigen
from hyperseti.dedoppler import dedoppler
from hyperseti.normalize import normalize
from hyperseti.data_array import DataArray
from hyperseti.kernels.smear_corr import apply_smear_corr

from hyperseti.log import get_logger
logger = get_logger('hyperseti.dedoppler', 'info')
from cupyx.scipy.ndimage import uniform_filter1d

def apply_boxcar_drift(data_array: DataArray):
    """ Apply boxcar filter to compensate for doppler smearing
    
    An optimal boxcar is applied per row of drift rate. This retrieves
    a sensitivity increase of sqrt(boxcar_size) for a smeared signal.
    (Still down a sqrt(boxcar_size) compared to no smearing case).
    
    Args:
        data_array (DataArray): Array to apply boxcar filters to
    
    Returns:
         data_array (DataArray): Array with boxcar filters applied.
    """
    logger.debug(f"apply_boxcar_drift: Applying moving average based on drift rate.")
    metadata = data_array.metadata
    
    # Note: dedoppler array no longer has time dimensions, so need to read
    # metadata stores in attributes (integration_time and n_integration)
    drates = cp.asarray(data_array.drift_rate.data)
    df = data_array.frequency.step.to('Hz').value
    dt = metadata['integration_time'].to('s').value / metadata['n_integration']

    # Compute smearing (array of n_channels smeared for given driftrate)
    smearing_nchan = cp.abs(dt * drates / df).astype('int32')
    smearing_nchan_max = cp.asnumpy(cp.max(smearing_nchan))

    # Apply boxcar filter to compensate for smearing
    boxcars = map(int, list(cp.asnumpy(cp.unique(smearing_nchan))))
    for boxcar_size in boxcars:
        idxs = cp.where(smearing_nchan == boxcar_size)
        # 1. uniform_filter1d computes mean. We want sum, so *= boxcar_size
        # 2. we want noise to stay the same, so divide by sqrt(boxcar_size)
        # combined 1 and 2 give aa sqrt(2) factor
        logger.debug(f"boxcar_size: {boxcar_size}, dedopp idxs: {idxs}")
        if boxcar_size > 1:
            data_array.data[idxs] = uniform_filter1d(data_array.data[idxs], size=boxcar_size, axis=2)
            data_array.data[idxs] *= np.sqrt(boxcar_size)
    return data_array

def test_smear_corr():
    metadata = {'fch1': 1000*u.MHz, 'dt': 10*u.s, 'df': 2.5*u.Hz}
    n_t = 128
    n_f = 2**12

    frame = stg.Frame(fchans=n_f*u.pixel,
                    tchans=n_t*u.pixel,
                    df=metadata['df'],
                    dt=metadata['dt'],
                    fch1=metadata['fch1'])
    MEAN, STD = 10, 1
    test_tones = [
    {'f_start': frame.get_frequency(index=n_f//2), 'drift_rate': -0.10*u.Hz/u.s, 'snr': 100, 'width': 2.5*u.Hz},
    ]

    noise = frame.add_noise(x_mean=MEAN, x_std=STD, noise_type='gaussian')

    for tone in test_tones:
        signal = frame.add_signal(stg.constant_path(f_start=tone['f_start'],
                                                drift_rate=tone['drift_rate']),
                            stg.constant_t_profile(level=frame.get_intensity(snr=tone['snr'])),
                            stg.gaussian_f_profile(width=tone['width']),
                            stg.constant_bp_profile(level=1))

    d = from_setigen(frame)
    d.data = cp.asarray(d.data)

    dd = dedoppler(d, max_dd=2, plan='optimal')
    idata = cp.asnumpy(dd.data).squeeze()

    dd_box = apply_boxcar_drift(dd)
    refdata = cp.asnumpy(dd_box.data).squeeze()

    dd = dedoppler(d, max_dd=2, plan='optimal')
    idata = cp.asnumpy(dd.data).squeeze()

    dd_smear = apply_smear_corr(dd)
    odata = cp.asnumpy(dd_smear.data).squeeze()

    assert np.allclose(odata[:, 10:-10], refdata[:, 10:-10])

if __name__ == "__main__":
    test_smear_corr()