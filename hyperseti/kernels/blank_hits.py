import cupy as cp

blank_hits_kernel = cp.RawKernel(r'''
extern "C" __global__
    __global__ void blankHitsKernel
        (const float *data, int64_t *cidx, int32_t *shift, int32_t N_pad, int32_t F, int32_t P, int32_t T, int32_t B)
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
        if (idx < B) {
          for (int t = 0; t < T; t++) {
            
            //     start_channel   +  timestep    + dedoppler offset
            idx  = cidx[tid]       + (F * t)      + (shift[tid] * t / T);

            if (idx < F * T && idx > 0 ) {  
                // blank hit
                data[idx] = 0;
                 
                // apply padding. Theoretically neighbouring hits can cause multiple writes
                // to the same memory index, but as we're setting to zero not an issue
                for (int p = 1; p < N_pad, p++) {
                   if ((idx - p) > 0) {
                      // TODO: This can still blank edge of previous spectra
                      data[idx - p] = 0; 
                    }
                    if ((idx + p) < F * T) {
                      data[idx + p] = 0; 
                    }    
                } // for int p
              } // if idx < F * T
            } for int t
        } // if idx < B  
    } // blankHitsKernel()

''', 'blankHitsKernel')
