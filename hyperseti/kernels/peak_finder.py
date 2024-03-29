import numpy as np
import cupy as cp

from .kernel_manager import KernelManager

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
    __syncthreads();
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
        if (K * tid < F) {
            maxval_k[tid] = maxval_in[K * tid];  // Set max to 0th entry within search box
            maxidx_f[tid] = K * tid;             // Assume 0th entry within search box is max
            maxidx_t[tid] = maxidx_in[K * tid];  // Grab the corresponding time index    
        }
               
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
    __syncthreads();
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


class PeakFinderMan(KernelManager):
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
        super().__init__('PeakFinder')
    
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
        self.workspace['maxval_gpu'] = cp.asarray(maxval)
        self.workspace['maxidx_gpu'] = cp.asarray(maxidx)
        self.workspace['maxval_k_gpu'] = cp.asarray(maxval_k)
        self.workspace['maxidx_f_gpu'] = cp.asarray(maxidx_f)
        self.workspace['maxidx_t_gpu'] = cp.asarray(maxidx_t)
    
    def execute(self, d_gpu: cp.ndarray):
        """ Execute peak finder """
        try:
            assert d_gpu.shape == (self.N_time, self.N_chan)
        except AssertionError:
            raise RuntimeError(f"Array dimensions {d_gpu.shape} do not match those passed during init() ({self.N_time}, {self.N_chan})")
        
        ws = self.workspace
        find_max_1D(d_gpu, ws['maxval_gpu'], ws['maxidx_gpu'])
        find_max_reduce(ws['maxval_gpu'], ws['maxidx_gpu'], ws['maxval_k_gpu'], 
                        ws['maxidx_f_gpu'], ws['maxidx_t_gpu'], self.K)

        return ws['maxval_k_gpu'], ws['maxidx_f_gpu'], ws['maxidx_t_gpu']
    
    def find_peaks(self, d_gpu: cp.ndarray, return_space: str='cpu'):
        """ Find peaks in data 
        
        Args:
            d_gpu (cp.ndarray): 2D data array to search
        """
        ws = self.workspace
        self.execute(d_gpu)
        if return_space != 'cpu':
            return ws['maxval_k_gpu'], ws['maxidx_f_gpu'], ws['maxidx_t_gpu']
        else:
            return cp.asnumpy(ws['maxval_k_gpu']), cp.asnumpy(ws['maxidx_f_gpu']), cp.asnumpy(ws['maxidx_t_gpu'])

    def hitsearch(self, d_arr: cp.ndarray, threshold: float, min_spacing: float, beam_id: int=0, return_space: str='cpu'):
        """ Find peaks in data above threshold

        Also applies a third-stage filter to ensure hits have a minimum spacing
        
        Args:
            d_gpu (cp.ndarray): (time, beam, freq) data array to search
        """
        if d_arr.shape[1] > 1:
            d_gpu = cp.copy(d_arr[:, beam_id])
        else:
            d_gpu = d_arr.squeeze()

        self.execute(d_gpu)

        mask  = self.workspace['maxval_k_gpu'] > threshold
        hits  = cp.asnumpy(self.workspace['maxval_k_gpu'][mask])
        idx_f = cp.asnumpy(self.workspace['maxidx_f_gpu'][mask])
        idx_t = cp.asnumpy(self.workspace['maxidx_t_gpu'][mask])

        if len(hits) < 1:
            return hits, idx_f, idx_t       # return empty lists
        else:
            df = np.column_stack((np.arange(len(hits)), hits, idx_f, idx_t))
            df = _group_hits(df, min_spacing)
            return df[:, 1], df[:, 2].astype('int32'), df[:, 3].astype('int32')

def _group_hits(df, min_spacing):
    """ Helper function to sort list of hits into groups 
    
    # Final stage of hitsearch: we need to make sure only one maxima within
    # The minimum spacing. We loop through and assign to groups
    # Then find the maximum for each group.

    # TODO: Speed up this code
    """
    groups = []
    cur = df[0]
    g = [cur, ]

    for row in df[1:]:
        if row[2] - cur[2] < min_spacing:
            g.append(row)
        else:
            groups.append(g)
            g = [row, ]
        cur = row
    groups.append(g)

    df = []
    for g in groups:
        if len(g) == 1:
            df.append(g[0])
        else:
            mv, mi = g[0][1], 0
            for i, h in enumerate(g[1:]):
                if mv < h[1]:
                    mv, mi = h[1], i + 1 
            df.append(g[mi])
    df = np.array(df)    
    return df

def peak_find(dedopp_data: cp.ndarray, threshold: float, min_spacing: float, beam_id: int=0, mm: PeakFinderMan=None):
        """ Find peaks in data above threshold

        Also applies a third-stage filter to ensure hits have a minimum spacing
        
        Args:
            dedopp_data (cp.ndarray): (dedopp, beam, freq) data array to search
        """
        K = 2**(int(np.log2(min_spacing)))
        N_chan=dedopp_data.shape[2]
        N_beam=dedopp_data.shape[1]
        N_time=dedopp_data.shape[0]    
        
        if isinstance(mm, PeakFinderMan):
            pf = mm
        else:
            pf = PeakFinderMan()
        pf.init(N_chan=N_chan, N_time=N_time, K=K)
        maxval_gpu, maxidx_f_gpu, maxidx_t_gpu = pf.hitsearch(dedopp_data, threshold, min_spacing, beam_id=beam_id)
        return maxval_gpu, maxidx_f_gpu, maxidx_t_gpu