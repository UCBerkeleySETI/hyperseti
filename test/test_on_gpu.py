import hyperseti
from hyperseti.utils import on_gpu
import numpy as np
import cupy as cp

import logbook
hyperseti.utils.logger.level  = logbook.DEBUG

@on_gpu
def _test_on_gpu(x, _rs='gpu'):
    if _rs == 'gpu':
        return cp.asarray(x)
    else:
        return cp.asnumpy(x)

@on_gpu
def _test_on_gpu2(xl, _rs='gpu'):
    if _rs == 'gpu':
        return cp.asarray(xl[0]), cp.asarray(xl[1]), 'I did not'
    else:
        return cp.asnumpy(xl[0]), cp.asnumpy(xl[1]), 'oh hai mark'

def test_on_gpu_decorator():
    test_data_cpu = np.zeros(shape=(32, 1, 512), dtype='float32')
    test_data_gpu = cp.asarray(test_data_cpu)
    test_data_list = [test_data_cpu, test_data_gpu]

    out = _test_on_gpu(test_data_cpu, _rs='gpu', return_space='cpu')
    assert isinstance(out, np.ndarray)
    out = _test_on_gpu(test_data_cpu, _rs='gpu', return_space='gpu')
    assert isinstance(out, cp.ndarray)
    out = _test_on_gpu(test_data_gpu, _rs='cpu', return_space='cpu')
    assert isinstance(out, np.ndarray)
    out = _test_on_gpu(test_data_gpu, _rs='cpu', return_space='gpu')
    assert isinstance(out, cp.ndarray)

    out = _test_on_gpu2(test_data_list, _rs='gpu', return_space='cpu')
    assert isinstance(out[0], np.ndarray)
    assert isinstance(out[1], np.ndarray)
    out = _test_on_gpu2(test_data_list, _rs='gpu', return_space='gpu')
    assert isinstance(out[0], cp.ndarray)
    assert isinstance(out[1], cp.ndarray)
    out = _test_on_gpu2(test_data_list, _rs='cpu', return_space='cpu')
    assert isinstance(out[0], np.ndarray)
    assert isinstance(out[1], np.ndarray)
    out = _test_on_gpu2(test_data_list, _rs='cpu', return_space='gpu')
    assert isinstance(out[0], cp.ndarray)
    assert isinstance(out[1], cp.ndarray)

if __name__ == "__main__":
    test_on_gpu_decorator()