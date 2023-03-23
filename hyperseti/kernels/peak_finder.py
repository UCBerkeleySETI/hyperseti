import numpy as np
import cupy as cp

# 1D search kernel along slow-varying axis
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
        maxval[tid] = 0;
        maxidx[tid] = 0;

        for (int t = 0; t < T; t++) {
            idx = tid + F * t;
            
            // Make sure we're not out of bounds 
            if (idx < F * T) {
                // Check if we have a new maxima
                if (data[idx] > maxval[tid]) {
                    //printf("data > curmax, tid %d idx %d", tid, idx);
                    maxval[tid]   = data[idx];
                    maxidx[tid] = t; 
                }
            }
        }
    }
''', 'maxKernel')

# Kernel to search within blocks of size K
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
        maxval_k[tid] = maxval_in[K * tid];  // Set max to 0th entry within search box
        maxidx_f[tid] = K * tid;             // Assume 0th entry within search box is max
        maxidx_t[tid] = maxidx_in[K * tid];  // Grab the corresponding time index                   

        for (int k = 0; k < K; k++) {
            idx = K * tid + k;

            // check we are within memory bounds
            if (idx < F) {
                if (maxval_in[idx] > maxval_k[tid]) {
                    maxval_k[tid] = maxval_in[idx];
                    maxidx_t[tid] = maxidx_in[idx]; 
                    maxidx_f[tid] = idx;
                }
            }           

        }
    }
''', 'maxReduceKernel')


def find_max_1D(d_gpu: cp.ndarray, maxval_gpu: cp.ndarray, maxidx_gpu: cp.ndarray):
    """ Run maxKernel -- finds maximum along slow-varying (outer) axis of 2D array

    Used by PeakFinder class, along with find_max_reduce second-stage search.
    
    Args:
        d_gpu (cp.ndarray): 2D array to run peak search on, e.g. (time, frequency)
        maxval_gpu (cp.ndarray): Max values found along slow-varying axis (e.g. time axis)
        maxidx_gpu (cp.ndarray): Corresponding index on slow-varying axis where max was found
    """
    # Setup grid and block dimensions
    N_time, N_chan = d_gpu.shape
    F_block = np.min((N_chan, 1024))
    F_grid  = N_chan // F_block
    #print(f"Kernel shape (grid, block) {(F_grid,), (F_block,)}")
    max_kernel((F_grid,), (F_block,), (d_gpu, maxval_gpu, maxidx_gpu, N_chan, N_time))


def find_max_reduce(maxval_gpu: cp.ndarray, maxidx_gpu: cp.ndarray, maxval_k_gpu: cp.ndarray, 
                    maxidx_f_gpu: cp.ndarray, maxidx_t_gpu: cp.ndarray, K: int):
    """ Run maxReduceKernel -- finds maximum for 1D array within blocked search kernel of size K
    
    Designed to process output of find_max_1D to do secondary search within blocks of size (N_time, K)

    Args:
        maxval_gpu (cp.ndarray): Max values output from find_max_1D
        maxidx_gpu (cp.ndarray): Max indexes output from find_max_1D 
        maxval_k_gpu (cp.ndarray): Array with N_chan // K elements, local maxima within block
        maxidx_f_gpu (cp.ndarray): Array with N_chan // K elements, max value falling within search block (e.g. frequency)
        maxidx_t_gpu (cp.ndarray): Array with N_chan // K elements, index of slowly-varying axis (e.g. time)
    """
    N_chan = maxval_gpu.shape[0]
    F_block = np.min((N_chan, 1024))
    F_grid = np.max((N_chan // F_block // K, 1))
    #print(f"Kernel shape (grid, block) {(F_grid,), (F_block,)}")
    max_reduce_kernel((F_grid,), (F_block,), 
                      (maxval_gpu, maxidx_gpu, maxval_k_gpu, maxidx_f_gpu, maxidx_t_gpu, N_chan, K))


class PeakFinder(object):
    """ Finds peaks in 2D arrays 

    Divides array up along fast-varying axis into blocks of size K, 
    then searches each block for the maximum value within the block.
    The search is done in two stages:
        1. Find maximum along the slow-varying axis (reduction by N_time)
        2. Find maximum in reduced array within search block of size K 
    
    The output is a list of maximum values and corresponding indexes, which
    is sorted by increasing frequency channel.

    Provides the following methods:
        pf.init()    - initialize/allocate memory
        pf.execute() - execute peak search

    Notes:
        This has not yet been tested on non 2^N sized arrays
        and will likely crash if N_chan % K != 0
    """
    def __init__(self):
        pass
    
    def init(self, N_chan: int, N_time: int, K: int):
        """ Initialize peak finder (allocate memory) 
        
        Args:
            N_chan (int): Number of channels in data array
            N_time (int): Number of time integrations
            K (int): Kernel search size 
        """
        
        # Initialize empty arrays
        maxval = np.zeros(N_chan, dtype='float32')
        maxidx = np.zeros(N_chan, dtype='int32')
        maxval_k = np.zeros(N_chan // K, dtype='float32')
        maxidx_f = np.zeros(N_chan // K, dtype='int32')
        maxidx_t = np.zeros(N_chan // K, dtype='int32')
        
        self.K = K
        self.N_chan = N_chan
        self.N_time = N_time
        
        # Allocate on GPU
        self.maxval_gpu = cp.asarray(maxval)
        self.maxidx_gpu = cp.asarray(maxidx)
        self.maxval_k_gpu = cp.asarray(maxval_k)
        self.maxidx_f_gpu = cp.asarray(maxidx_f)
        self.maxidx_t_gpu = cp.asarray(maxidx_t)
        
    
    def execute(self, d_gpu: cp.ndarray):
        """ Execute peak finder """
        try:
            assert d_gpu.shape == (self.N_time, self.N_chan)
        except AssertionError:
            raise RuntimeError("Array dimensions do not match those passed during init()")
        find_max_1D(d_gpu, self.maxval_gpu, self.maxidx_gpu)
        find_max_reduce(self.maxval_gpu, self.maxidx_gpu, self.maxval_k_gpu, 
                        self.maxidx_f_gpu, self.maxidx_t_gpu, self.K)
        return self.maxval_k_gpu, self.maxidx_f_gpu, self.maxidx_t_gpu
    
    def find_peaks(self, d_gpu: cp.ndarray, return_space: str='cpu'):
        """ Find peaks in data 
        
        Args:
            d_gpu (cp.ndarray): 2D data array to search
        """
        self.execute(d_gpu)
        if return_space != 'cpu':
            return self.maxval_k_gpu, self.maxidx_f_gpu, self.maxidx_t_gpu
        else:
            return cp.asnumpy(self.maxval_k_gpu), cp.asnumpy(self.maxidx_f_gpu), cp.asnumpy(self.maxidx_t_gpu)

    def hitsearch(self, d_arr: cp.ndarray, threshold: float, min_spacing: float, beam_id: int=0, return_space: str='cpu'):
        """ Find peaks in data above threshold

        Also applies a third-stage filter to ensure hits have a minimum spacing
        
        Args:
            d_gpu (cp.ndarray): 2D data array to search
        """
        if d_arr.shape[1] > 1:
            d_gpu = cp.copy(d_arr[:, beam_id])
        else:
            d_gpu = d_arr.squeeze()

        self.execute(d_gpu)

        mask  = self.maxval_k_gpu > threshold
        hits  = self.maxval_k_gpu[mask]
        idx_f = self.maxidx_f_gpu[mask]
        idx_t = self.maxidx_t_gpu[mask]

        if len(hits) >= 1e9:
            # Now we want to look for any hits that are spaced by < K
            # (This is possible if hits are near edge of search space)
            f_spacing = cp.diff(idx_f)
            mask = f_spacing > min_spacing
            mask = cp.concatenate((cp.asarray([True, ]), mask))
            hits = hits[mask]
            idx_f = idx_f[mask]
            idx_t = idx_t[mask]

        if return_space == 'cpu':
            return cp.asnumpy(hits), cp.asnumpy(idx_f), cp.asnumpy(idx_t)
        else:
            return hits, idx_f, idx_t


    def __del__(self):
        """ Free memory when deleted 
        
        See https://docs.cupy.dev/en/stable/user_guide/memory.html
        """
        mempool = cp.get_default_memory_pool()
        self.maxval_gpu   = None
        self.maxidx_gpu   = None
        self.maxval_k_gpu = None
        self.maxidx_f_gpu = None
        self.maxidx_t_gpu = None
        mempool.free_all_blocks()