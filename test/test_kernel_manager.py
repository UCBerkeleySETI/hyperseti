from hyperseti.kernels.kernel_manager import KernelManager
from hyperseti.kernels.smear_corr import SmearCorrMan

def test_kernel_manager():
    dd_shape = (2049, 1, 4096)
    sc = SmearCorrMan()
    sc.init(*dd_shape)
    print(sc.info())

if __name__ == "__main__":
    test_kernel_manager()