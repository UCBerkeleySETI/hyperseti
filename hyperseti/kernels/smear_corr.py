import cupy as cp
import numpy as np

from .kernel_manager import KernelManager
from ..data_array import DataArray

smear_corr_kernel = cp.RawKernel(r'''
extern "C" __global__
    __global__ void smearCorrKernel
        (float *idata, float *odata, int *N_chan_smear, int F, int D)
        /* Each thread computes a different dedoppler sum for a given channel
         
         F: N_frequency channels
         D: N_dedopp steps
        
         *idata: Data array, (D x F) shape
         *odata: Data array, (D x F) shape
         *N_chan_smear: Number of channels smearing (one entry per timestep, Dx1 vector)
        */
        {
        
        // Setup thread index
        // Thread index == frequency index
        const int f = blockIdx.x * blockDim.x + threadIdx.x;  
        const int osize = F * D;

        int N_smear = 1;
        int idx     = 0;
        
        if (f < F) {
          for (int d = 0; d < D; d++) {
            // idx = start_channel   +  timestep   
            idx    = f               + (F * d); 

            // Get N_smear from input vector
            N_smear = N_chan_smear[d];
            
            if (N_smear > 1) {
                // Do the moving average on each data point
                if (f + N_smear/2 < F && f - N_smear/2 > 0) {
                    float movsum = 0;
                    for (int i = 0; i < N_smear; i++) {
                        movsum += idata[idx+i-N_smear/2];
                    }   
                    odata[idx] = movsum / sqrt((float)N_smear);
                } else {
                    odata[idx] = idata[idx];
                }
            } else { 
                odata[idx] = idata[idx];
            }  
          }
        }
    }

''', 'smearCorrKernel')


class SmearCorrMan(KernelManager):
    """ Kernel manager for smearing correction """
    def __init__(self):
        super().__init__('SmearCorrMan')
    
    def init(self, N_dedopp: int, N_beam: int, N_chan: int):
        """ Initialize (or reinitialize) kernel 
        
        Args:
            N_dedopp (int): Number of dedoppler trials in input data
            N_beam (int): Number of beams in input data
            N_chan (int): Number of frequency channels
        """
        self.N_chan = N_chan
        self.N_beam  = N_beam
        self.N_dedopp  = N_dedopp

        odata = np.zeros(shape=(N_dedopp, N_beam, N_chan), dtype='float32')
        self.workspace['odata'] = cp.asarray(odata)

        N_grid = np.min((N_chan, 1024))
        self._grid  = (N_grid, )
        self._block = (N_chan // self._grid[0], ) 

    def execute(self, dedopp_array: DataArray) -> DataArray:
        """ Execute kernel on dedoppler data array """
        drates = cp.asarray(dedopp_array.drift_rate.data)
        df = dedopp_array.frequency.step.to('Hz').value
        dt = dedopp_array.metadata['integration_time'].to('s').value / dedopp_array.metadata['n_integration']
        smearing_nchan = cp.abs(dt * drates / df).astype('int32')
        smearing_nchan_gpu = cp.asarray(smearing_nchan)

        smear_corr_kernel(self._grid, self._block, (dedopp_array.data, self.workspace['odata'], smearing_nchan_gpu, self.N_chan, self.N_dedopp))
        dedopp_array.data = self.workspace['odata']
        return dedopp_array


def apply_smear_corr(dedopp_array: DataArray) -> DataArray:
    """ Apply smearing correction 

    An optimal boxcar is applied per row of drift rate. This retrieves
    a sensitivity increase of sqrt(boxcar_size) for a smeared signal.
    (Still down a sqrt(boxcar_size) compared to no smearing case).
    
    Args:
        data_array (DataArray): Array to apply boxcar filters to
    
    Returns:
         data_array (DataArray): Array with boxcar filters applied.

    Notes:
        This is a GPU kernel version of dedoppler.apply_boxcar_drift
        This will not work for N_b > 1 at the moment!
    """
    sc = SmearCorrMan()
    sc.init(*dedopp_array.shape)
    dedopp_array = sc.execute(dedopp_array)
    return dedopp_array
