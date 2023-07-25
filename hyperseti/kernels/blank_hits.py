import cupy as cp
import numpy as np
from .kernel_manager import KernelManager

from ..log import get_logger
logger = get_logger('hyperseti.blanking')

blank_hits_kernel = cp.RawKernel(r'''
extern "C" __global__
    __global__ void blankHitsKernel
        (float *data, int *cidx, int *shift, int *N_pad_lower, int *N_pad_upper, int F, int P, int T, int B)
        /* Each thread computes a different dedoppler sum for a given channel
         
         N_pad: Padding either side of hit
         F: N_frequency channels
         T: N_time steps
         P: N_pol (TODO: Acutally implement pol support!)
         B: N_blank
        
         *data: Data array, (T x F) shape
         *cidx: Array of frequency channel index at start t=0
         *shift: Array of doppler corrections of length D.
                 shift is total number of channels to shift at time t
        */
        {
        
        // Setup thread index
        // Thread index == dedoppler trial index
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;  

        // Index for output array
        
        int idx = 0;
        //printf("TID %d \n", tid);
        if (tid < B) {
          for (int t = 0; t < T; t++) {
            
            //     start_channel   +  timestep    + dedoppler offset
            idx  = cidx[tid]       + (F * t)      + (shift[tid] * t);
            //printf("TID %d IDX %d\n", tid, idx);
            
            if (idx < F * T && idx > 0 ) {  
                // blank hit
                data[idx] = 0;
                 
                // apply padding. Theoretically neighbouring hits can cause multiple writes
                // to the same memory index, but as we're setting to zero not an issue
                int N_up = N_pad_upper[tid];
                int N_lo = N_pad_lower[tid];

                for (int p = 1; p < N_up; p++) {
                    if (idx + p < F * T) {
                      data[idx + p] = 0.0; 
                    }
                  }
                for (int p = -1; p > N_lo; p--) {   
                   if (idx + p > 0) {
                      // TODO: This can still blank edge of previous spectra
                      data[idx + p] = 0.0; 
                    }
                  }

              } // if idx < F * T
            } // for int t
        } // if idx < B  
    } // blankHitsKernel()

''', 'blankHitsKernel')

class BlankHitsMan(KernelManager):
    def __init__(self):
        super().__init__('BlankHitsMan')
        self.N_chan    = None
        self.N_beam    = None
        self.N_time    = None
        self.N_blank   = None
    
    def init(self, N_time: int, N_beam: int, N_chan: int, N_blank: int):
        """ Initialize (or reinitialize) kernel 
        
        Args:
            N_time (int): Number of timesteps in input data
            N_beam (int): Number of beams in input data
            N_chan (int): Number of frequency channels
        """
        reinit = False
        if N_chan != self.N_chan: reinit = True
        if N_beam != self.N_beam: reinit = True
        if N_time != self.N_time: reinit = True
        if N_blank != self.N_blank: reinit = True

        if reinit:
            logger.debug(f"BlankHitsMan: Reinitializing")
            self.N_chan    = N_chan
            self.N_beam    = N_beam
            self.N_time    = N_time
            self.N_blank   = N_blank

            N_threads = np.min((N_blank, 1024))
            N_grid    = N_blank // N_threads
            if N_blank % N_threads != 0:
                N_grid += 1
            logger.debug(f"BlankHitsMan: Kernel shape (grid, block) {(N_grid, ), (N_threads,)}")

            self._grid  = (N_grid, 1, 1)
            self._block = (N_threads, 1, 1)

    def execute(self, d_gpu: cp.ndarray, cidxs_gpu: cp.ndarray, dd_shift_gpu: cp.ndarray, 
                N_pad_lower: cp.ndarray, N_pad_upper: cp.ndarray):
        """ Execute kernel on dedoppler data array """
        blank_hits_kernel(self._grid, self._block,
                         (d_gpu, cidxs_gpu, dd_shift_gpu, N_pad_lower, N_pad_upper, 
                          self.N_chan, self.N_beam, self.N_time, self.N_blank)) 
