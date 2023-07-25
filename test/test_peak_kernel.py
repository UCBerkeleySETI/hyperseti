import numpy as np
import cupy as cp
from hyperseti.kernels.peak_finder import PeakFinderMan, peak_find
from hyperseti.normalize import normalize
from hyperseti.io import from_h5

from hyperseti.test_data import voyager_h5

def find_max_np(a, K):
    """ numpy-based kernel for testing """
    m = np.max(a, axis=0)
    m_idx = np.argmax(a, axis=0)
    m, m_idx

    M         = np.max(m.reshape((-1, K)), axis=1)
    M_offset  = K * np.arange(m.shape[0] // K)
    M_k_idx   = np.argmax(m.reshape((-1, K)), axis=1) 
    M_f_idx   = M_k_idx + M_offset
    M_t_idx   = m_idx[M_k_idx + M_offset] 
    
    return M, M_f_idx, M_t_idx

def find_max_cp(a, K):
    """ Cupy version of numpy kernel """
    a = cp.asarray(a)
    m = cp.max(a, axis=0)
    m_idx = cp.argmax(a, axis=0)
    m, m_idx

    M         = cp.max(m.reshape((-1, K)), axis=1)
    M_offset  = K * cp.arange(m.shape[0] // K)
    M_k_idx   = cp.argmax(m.reshape((-1, K)), axis=1)
    M_f_idx   = M_k_idx + M_offset
    M_t_idx   = m_idx[M_k_idx + M_offset] 
    
    return cp.asnumpy(M), cp.asnumpy(M_f_idx), cp.asnumpy(M_t_idx)


def test_peak_kernel():
    # A simple test array for K=4 kernel
    # Maxvals are [1 2 3 4]
    # Max f_idxs  [0 5 10 15]
    # Max t_idx   [1 0 1 0]
    a = np.array(
        [[0,0,0,0, 0,2,0,0, 0,0,0,0, 0,0,0,4],
        [1,0,0,0,  0,0,0,0, 0,0,3,0, 0,0,0,0]], 
        dtype='float32')
    a_gpu = cp.asarray(a)
    N_chan, N_time, K = 16, 2, 4

    # Test numpy
    maxval_cpu, maxidx_f_cpu, maxidx_t_cpu = find_max_np(a, K)
    print(maxval_cpu, maxidx_f_cpu, maxidx_t_cpu)

    assert np.allclose(maxval_cpu, (1,2,3,4))
    assert np.allclose(maxidx_f_cpu, (0,5,10,15))
    assert np.allclose(maxidx_t_cpu, (1,0,1,0))

    # Test cupy
    maxval_gpu, maxidx_f_gpu, maxidx_t_gpu = find_max_cp(a, K)
    print(maxval_gpu, maxidx_f_gpu, maxidx_t_gpu)

    assert np.allclose(maxval_gpu, (1,2,3,4))
    assert np.allclose(maxidx_f_gpu, (0,5,10,15))
    assert np.allclose(maxidx_t_gpu, (1,0,1,0))

    # Test peak_kernel
    pf = PeakFinderMan()
    pf.init(N_chan=16, N_time=2, K=4)
    print(pf.info())
    maxval_gpu, maxidx_f_gpu, maxidx_t_gpu = pf.find_peaks(a_gpu)

    print(maxval_gpu, maxidx_f_gpu, maxidx_t_gpu)

    assert np.allclose(maxval_gpu, (1,2,3,4))
    assert np.allclose(maxidx_f_gpu, (0,5,10,15))
    assert np.allclose(maxidx_t_gpu, (1,0,1,0))

    ## Now test with larger array
    b = np.zeros((2, 2**20), dtype='float32')
    b[0, :-10:10]  = np.arange(1, b.shape[1] // 10 * 2, 2)
    b[1, 5:-10:10] = np.arange(1, b.shape[1] // 10 * 2, 2) + 1
    b_gpu = cp.asarray(b)

    K = 4
    pf = PeakFinderMan()
    pf.init(N_chan=b.shape[1], N_time=b.shape[0], K=K)
    print(pf.info())
    maxval_gpu, maxidx_f_gpu, maxidx_t_gpu = pf.find_peaks(b_gpu, return_space='cpu')

    maxval_cpu, maxidx_f_cpu, maxidx_t_cpu = find_max_np(b, K)

    print(maxval_gpu.shape, maxval_cpu.shape)
    #print(maxval_gpu[:100])
    #print(maxval_cpu[:100])
    print("freq idx")
    print(maxidx_f_gpu[:104])
    print(maxidx_f_cpu[:104])

    assert np.allclose(maxval_cpu, maxval_gpu)
    assert np.allclose(maxidx_f_cpu, maxidx_f_gpu)
    assert np.allclose(maxidx_t_cpu, maxidx_t_gpu)

    return  maxval_gpu, maxidx_f_gpu, maxidx_t_gpu 

def test_resize():
    
    N_timesteps = (2, 16, 32, 59, 4, 32)
    N_channels  = (2**18, 2**19, 2**20, 2**19)
    N_iter      = 3
    N_pol       = 1
    threshold   = 10

    for ii in range(N_iter):
        for N_time in N_timesteps:
            for N_chan in N_channels:
                print(f"({ii+1}/{N_iter}) Data shape: ({N_time}, {N_pol}, {N_chan})")

                b = np.zeros(N_time *  N_chan, dtype='float32').reshape((N_time, N_pol, N_chan))

                b[N_time - 1, 0, N_chan // 2 + 1] = 100
                b[N_time - 1, 0, N_chan // 2 - 1000] = 50
                #print(b.shape)
                b_gpu = cp.asarray(b)
                K = 64
                pf = PeakFinderMan()
                pf.init(N_chan=b_gpu.shape[2], N_time=b_gpu.shape[0], K=K)
                maxval_gpu, maxidx_f_gpu, maxidx_t_gpu = pf.hitsearch(b_gpu, threshold=10, min_spacing=100, return_space='cpu')
                maxval_cpu, maxidx_f_cpu, maxidx_t_cpu = cp.asnumpy(maxval_gpu),  cp.asnumpy(maxidx_f_gpu), cp.asnumpy(maxidx_t_gpu)


def test_hitsearch():
    v = from_h5(voyager_h5)
    vs = v.sel({'frequency': slice(0, 2**20)})
    vs.data = cp.asarray(vs.data)
    vs = normalize(vs, poly_fit=5)

    pf = PeakFinderMan()
    pf.init(N_chan=vs.frequency.n_step, N_time=vs.time.n_step, K=32)
    mv, f_idx, t_idx = pf.hitsearch(vs.data, threshold=3, min_spacing=100)
    print(mv, f_idx, t_idx)

    # Test manual peak_find
    peak_find(vs.data, threshold=3, min_spacing=100, mm=pf)
    peak_find(vs.data, threshold=3, min_spacing=100)
    print(pf.info())



if __name__ == "__main__":
     #test_peak_kernel()
     test_hitsearch()
     #test_resize()