from hyperseti.hits import get_signal_extent, get_signal_extents
import cupy as cp

def test_extent():
    d = cp.zeros(shape=(100, 1, 2000))

    N_x = 15
    d[:, 0, 505-N_x:505+N_x] = 100
    d0, p0, g0 = 50, 0, 505
    
    el, eu = get_signal_extent(d, d0, p0, g0)
    assert abs(el + N_x) <= 1
    assert abs(eu - N_x) <= 1

    d0, p0, g0 = 50, 0, 1500
    N_x = 50
    d[:, 0, 1500-N_x:1500+N_x] = 100
    el, eu = get_signal_extent(d, d0, p0, g0)
    assert abs(el + N_x) <= 1
    assert abs(eu - N_x) <= 1



if __name__ == "__main__":
    test_extent()