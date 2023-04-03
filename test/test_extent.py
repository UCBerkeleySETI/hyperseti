from hyperseti.hits import get_signal_extent, get_signal_extents
import cupy as cp

def test_extent():
    d = cp.zeros(shape=(100, 1, 2000))
    d[:, 0, 505-15:505+15] = 100

    d[:, 0, 1500-20:1500+50+1] = 100

    d0, p0, g0 = 50, 0, 505
    assert get_signal_extent(d, d0, p0, g0) == (-16, 16)

    d0, p0, g0 = 50, 0, 1500
    assert get_signal_extent(d, d0, p0, g0) == (-32, 64)

def test_extents():
    

if __name__ == "__main__":
    test_extent()