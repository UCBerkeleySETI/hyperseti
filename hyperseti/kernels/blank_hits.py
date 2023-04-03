import cupy as cp

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
                    if (idx < F * T - p) {
                      data[idx + p] = 0.0; 
                    }
                  }
                for (int p = -1; p > N_lo; p--) {   
                   if (idx > p) {
                      // TODO: This can still blank edge of previous spectra
                      data[idx + p] = 0.0; 
                    }
                  }

              } // if idx < F * T
            } // for int t
        } // if idx < B  
    } // blankHitsKernel()

''', 'blankHitsKernel')
