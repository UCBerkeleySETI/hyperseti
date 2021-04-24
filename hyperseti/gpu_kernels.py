import cupy as cp

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
              dedopp[dd_idx] = dd_val;
            }
        }
''', 'dedopplerKernel')


dedoppler_kurtosis_kernel = cp.RawKernel(r'''
extern "C" __global__
    __global__ void dedopplerKurtosisKernel
        (const float *data, float *dedopp, int *shift, int F, int T, int N)
        /* Each thread computes a different dedoppler sum for a given channel
        
         F: N_frequency channels
         T: N_time steps
         N: N_acc number of accumulations averaged within time step
         
         *data: Data array, (T x F) shape
         *dedopp: Dedoppler summed data, (D x F) shape
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