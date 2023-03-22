import numpy as np
import cupy as cp

max_kernel = cp.RawKernel(r'''
extern "C" __global__
    __global__ void maxKernel
        (const float *data, float *maxval, int *maxidx, int F, int T)
        /* Each thread computes a different dedoppler sum for a given channel
        
         F: N_frequency channels
         T: N_timesteps

         * maxval:   Peak values array
         * maxidx:  Peak T index array
        */
        {
        
        // Setup thread index
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        int idx = 0;
        maxval[tid]   = data[tid];
        maxidx[tid] = 0;

        for (int t = 1; t < T; t++) {
            idx = tid + F * t;

            if (data[idx] > maxval[tid]) {
              maxval[tid]   = data[idx];
              maxidx[tid] = t; 
              }
            }
        }
''', 'maxKernel')

max_reduce_kernel = cp.RawKernel(r'''
extern "C" __global__
    __global__ void maxReduceKernel
        (const float *maxval_in, int *maxidx_in, float *maxval_k, int *maxidx_f, int *maxidx_t, int F, int K)
        /* Each thread finds maxidx and maxval within kernel search footprint
        
         F: N_frequency channels
         K: search kernel size

         * maxval_k:   Peak values array
         * maxidx_f:  Peak F index array
         * maxidx_t:  Peak T index array
        */
        {
        
        // Setup thread index
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        int idx = 0;
        maxval_k[tid] = 0;
        maxidx_f[tid] = 0;
        maxidx_t[tid] = 0;

        for (int k = 0; k < K; k++) {
            idx = tid + K * k;
            if(idx < F && tid < F / K) {
               if (maxval_in[idx] > maxval_k[tid]) {
                  //intf("tid %d idx %d", tid, idx);
                  maxval_k[tid] = maxval_in[idx];
                  maxidx_t[tid] = maxidx_in[idx]; 
                  maxidx_f[tid] = idx;
                }
            }
        }
    }
''', 'maxReduceKernel')

def find_max_cpu(d_cpu):
    maxval_cpu = np.max(d, axis=0)
    maxidx_cpu = np.argmax(d, axis=0)
    return maxval_cpu, maxidx_cpu

def find_max_cupy(d_gpu):
    maxval_gpu = cp.max(d_gpu, axis=0)
    maxidx_gpu = cp.argmax(d_gpu, axis=0)
    return maxval_gpu, maxidx_gpu

def find_max_1D(d_gpu, maxval_gpu, maxidx_gpu):
    # Setup grid and block dimensions
    N_time, N_chan = d_gpu.shape
    F_block = np.min((N_chan, 1024))
    F_grid  = N_chan // F_block
    print(f"Kernel shape (grid, block) {(F_grid,), (F_block,)}")
    max_kernel((F_grid,), (F_block,), (d_gpu, maxval_gpu, maxidx_gpu, N_chan, N_time))

def find_max_reduce(maxval_gpu, maxidx_gpu, maxval_k_gpu, maxidx_f_gpu, maxidx_t_gpu, K):
    N_chan = maxval_gpu.shape[0]
    F_block = np.min((N_chan, 1024))
    F_grid = np.max((N_chan // F_block // K, 1))
    print(f"Kernel shape (grid, block) {(F_grid,), (F_block,)}")
    max_reduce_kernel((F_grid,), (F_block,), 
                      (maxval_gpu, maxidx_gpu, maxval_k_gpu, maxidx_f_gpu, maxidx_t_gpu, N_chan, K))

def find_max(d, K):
    maxval = np.zeros(N_chan, dtype='float32')
    maxidx = np.zeros(N_chan, dtype='int32')

    d_gpu = cp.asarray(d)
    maxval_gpu = cp.asarray(maxval)
    maxidx_gpu = cp.asarray(maxidx)

    maxval_k = np.zeros(N_chan // K, dtype='float32')
    maxidx_f = np.zeros(N_chan // K, dtype='int32')
    maxidx_t = np.zeros(N_chan // K, dtype='int32')

    maxval_k_gpu = cp.asarray(maxval_k)
    maxidx_f_gpu = cp.asarray(maxidx_f)
    maxidx_t_gpu = cp.asarray(maxidx_t)

    find_max_1D(d_gpu, maxval_gpu, maxidx_gpu)
    find_max_reduce(maxval_gpu, maxidx_gpu, maxval_k_gpu, maxidx_f_gpu, maxidx_t_gpu, K)
    return maxval_k_gpu, maxidx_f_gpu, maxidx_t_gpu