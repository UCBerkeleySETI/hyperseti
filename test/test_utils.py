import cupy as cp
from hyperseti.utils import attach_gpu_device, timeme
from hyperseti.log import get_logger

@timeme
def time_this_fn():
    import time
    time.sleep(1)

def test_timeme():
    tlog = get_logger('hyperseti.timer', 'info')
    time_this_fn()

def test_attach_gpu():
    ulog = get_logger('hyperseti.utils', 'info')
    attach_gpu_device(0)
    attach_gpu_device(0)
    if cp.cuda.runtime.getDeviceCount() > 1:
        attach_gpu_device(1)
        attach_gpu_device(1)

if __name__ == "__main__":
    test_timeme()
    test_attach_gpu()