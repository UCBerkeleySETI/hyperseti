import cupy as cp
import numpy as np

from .kernel_manager import KernelManager
from ..log import get_logger
from ..filter import apply_boxcar
from ..data_array import DataArray

logger = get_logger('hyperseti.dedoppler')

dedoppler_kernel = cp.RawKernel(r'''
extern "C" __global__
    __global__ void dedopplerKernel
        (const float *data, float *dedopp, int *shift, int F, int T)
        /* Each thread computes a different dedoppler sum for a given channel
        
         F: N_frequency channels
         T: N_time steps
        
         *data: Data array, (T x F) shape
         *dedopp: Dedoppler summed data, (D x F) shape
         *shift: Array of doppler corrections of length D.
                 shift is total number of channels to shift at time T
        */
        {
        
        // Setup thread index
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int d   = blockIdx.y;   // Dedoppler trial ID
        const int D   = gridDim.y;   // Number of dedoppler trials
        // Index for output array
        const int dd_idx = d * F + tid;
        float dd_val = 0;
        
        int idx = 0;
        for (int t = 0; t < T; t++) {
                            // timestep    // dedoppler trial offset
            idx  = tid + (F * t)      + (shift[d] * t / T);
            if (idx < F * T && idx > 0) {
                dd_val += data[idx];
              }
              // Divide through by sqrt(T) to keep S/N same as input
              float Tf = (float)T;
              dedopp[dd_idx] = dd_val / sqrt(Tf);
            }
        }
''', 'dedopplerKernel')


dedoppler_kurtosis_kernel = cp.RawKernel(r'''
extern "C" __global__
    __global__ void dedopplerKurtosisKernel
        (const float *data, float *dedopp, int *shift, int F, int T, int N)
        /* Each thread computes a different dedoppler SK for a given channel
        
         F: N_frequency channels
         T: N_time steps
         N: N_acc number of accumulations averaged within time step
         
         *data: Data array, (T x F) shape
         *dedopp: Dedoppler spectralkurtosis data, (D x F) shape
         *shift: Array of doppler corrections of length D.
                 shift is total number of channels to shift at time T
        
        Note: output needs to be scaled by N_acc, number of time accumulations
        */
        {
        
        // Setup thread index
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int d   = blockIdx.y;   // Dedoppler trial ID
        const int D   = gridDim.y;   // Number of dedoppler trials

        // Index for output array
        const int dd_idx = d * F + tid;
        float S1 = 0;
        float S2 = 0;
        
        int idx = 0;
        for (int t = 0; t < T; t++) {
                            // timestep    // dedoppler trial offset
            idx  = tid + (F * t)      + (shift[d] * t / T);
            if (idx < F * T && idx > 0) {
                S1 += data[idx];
                S2 += data[idx] * data[idx];
              }
              dedopp[dd_idx] = (N*T+1)/(T-1) * (T*(S2 / (S1*S1)) - 1);
            }
        }
''', 'dedopplerKurtosisKernel')

dedoppler_with_kurtosis_kernel = cp.RawKernel(r'''
extern "C" __global__
    __global__ void dedopplerWithKurtosisKernel
        (const float *data, float *dedopp, float *dedopp_sk, int *shift, int F, int T, int N)
        /* Each thread computes a different dedoppler sum and DDSK for a given channel
        
         F: N_frequency channels
         T: N_time steps
         N: N_acc number of accumulations averaged within time step
         
         *data: Data array, (T x F) shape
         *dedopp: Dedoppler summed data, (D x F) shape
         *dedopp_sk: Dedoppler spectral kurtosis data, (D x F) shape
         *shift: Array of doppler corrections of length D.
                 shift is total number of channels to shift at time T
        */
        {
        
        // Setup thread index
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int d   = blockIdx.y;   // Dedoppler trial ID
        const int D   = gridDim.y;   // Number of dedoppler trials

        // Index for output array
        const int dd_idx = d * F + tid;
        float S1 = 0;
        float S2 = 0;
        float Tf = (float)T;
        
        int idx = 0;
        for (int t = 0; t < T; t++) {
                            // timestep    // dedoppler trial offset
            idx  = tid + (F * t)      + (shift[d] * t / T);
            if (idx < F * T && idx > 0) {
                S1 += data[idx];
                S2 += data[idx] * data[idx];
              }
              dedopp[dd_idx] = S1 / sqrt(Tf);
              dedopp_sk[dd_idx] = (N*T+1)/(T-1) * (T*(S2 / (S1*S1)) - 1);
            }
        }

''', 'dedopplerWithKurtosisKernel')


class DedopplerMan(KernelManager):
    """ Kernel manager for smearing correction """
    def __init__(self):
        super().__init__('DedopplerMan')
        self.N_time  = None
        self.N_chan  = None
        self.N_beam  = None
        self.N_dopp  = None
        self.kernel  = None
    
    def init(self, N_time: int, N_beam: int, N_chan: int, N_dopp: int, kernel='dedoppler'):
        """ Initialize (or reinitialize) kernel 
        
        Args:
            N_dopp (int): Number of dedoppler trials in input data
            N_chan (int): Number of frequency channels
        """

        reinit = False
        if N_time != self.N_time: reinit = True
        if N_chan != self.N_chan: reinit = True
        if N_beam != self.N_beam: reinit = True
        if N_dopp != self.N_dopp: reinit = True
        if kernel != self.kernel: reinit = True

        if reinit:
            logger.debug(f'DedopplerMan: Reinitializing')
            self.N_time = N_time
            self.N_chan = N_chan
            self.N_beam  = N_beam
            self.N_dopp  = N_dopp

            self.kernel = kernel

            # Allocate GPU memory for dedoppler data
            self.workspace['dedopp'] = cp.zeros((N_dopp, N_beam, N_chan), dtype=cp.float32)
            if self.kernel == 'ddsk':
                self.workspace['dedopp_sk'] =  cp.zeros((N_dopp, N_beam, N_chan), dtype=cp.float32)

            if N_beam > 1:
                self.workspace['_dedopp'] = cp.zeros((N_dopp, N_chan), dtype=cp.float32)
                if self.kernel == 'ddsk':
                    self.workspace['_dedopp_sk'] =  cp.zeros((N_dopp, N_chan), dtype=cp.float32)
            else:
                self.workspace['_dedopp'] = self.workspace['dedopp']
                if self.kernel == 'ddsk':
                    self.workspace['_dedopp_sk'] =  self.workspace['dedopp_sk']

            # Setup grid and block dimensions
            F_block = np.min((N_chan, 1024))
            F_grid  = N_chan // F_block
            self._grid = (F_grid, N_dopp)
            self._block = (F_block,)


    def execute(self, data_array: DataArray, dd_shifts_gpu: cp.ndarray, boxcar_size: int=1) -> DataArray:
        """ Execute kernel to compute dedoppler array """
        ws = self.workspace
        # Calculate number of integrations within each (time-averaged) channel
        samps_per_sec = np.abs((1.0 / data_array.frequency.step).to('s').value) / 2 # Nyq sample rate for channel
        N_acc = int(data_array.time.step.to('s').value / samps_per_sec)
        
        # TODO: Candidate for parallelization
        for beam_id in range(self.N_beam):

            # Select out beam
            d_gpu = data_array.data[:, beam_id, :] 

            # Apply boxcar filter
            if boxcar_size > 1:
                d_gpu = apply_boxcar(d_gpu, boxcar_size=boxcar_size, mode='gaussian')
            
            if self.kernel == 'dedoppler':
                logger.debug(f"{type(d_gpu)}, {type(ws['dedopp'])}, {self.N_chan}, {self.N_time}")
                dedoppler_kernel(self._grid, self._block, 
                                (d_gpu, ws['_dedopp'], dd_shifts_gpu, self.N_chan, self.N_time)) # grid, block and arguments
            elif self.kernel == 'kurtosis':
                # output must be scaled by N_acc, which can be figured out from df and dt metadata
                logger.debug(f'dedoppler kurtosis: rescaling SK by {N_acc}')
                logger.debug(f"dedoppler kurtosis: driftrates: {dd_shifts_gpu}")
                dedoppler_kurtosis_kernel(self._grid, self._block,
                                (d_gpu, ws['_dedopp'], dd_shifts_gpu, self.N_chan, self.N_time, N_acc)) # grid, block and arguments 
            elif self.kernel == 'ddsk':
                # output must be scaled by N_acc, which can be figured out from df and dt metadata
                logger.debug(f'dedoppler ddsk: rescaling SK by {N_acc}')
                logger.debug(f"dedoppler ddsk: driftrates: {dd_shifts_gpu}")
                dedoppler_with_kurtosis_kernel(self._grid, self._block,
                                (d_gpu, ws['_dedopp'], ws['_dedopp_sk'], dd_shifts_gpu, self.N_chan, self.N_time, N_acc)) 
            else:
                logger.critical("dedoppler: Unknown kernel={} !!".format(kernel))
                raise RuntimeError("Dedoppler failed, unknown kernel!")
      
            if self.N_beam > 1:
                ws['dedopp'][:, beam_id] = ws['_dedopp']
                if self.kernel == 'ddsk':
                    ws['dedopp_sk'][:, beam_id] = ws['_dedopp_sk']

        if self.kernel == 'ddsk':
          return ws['dedopp'], ws['dedopp_sk']
        else:
          return ws['dedopp']