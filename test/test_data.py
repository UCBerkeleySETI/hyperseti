from hyperseti.data_array import from_fil, from_h5
from file_defs import voyager_fil, voyager_h5

def test_load_voyager():
    d = from_fil(voyager_fil)

    print(d)
    print(d.attrs)
    print(d.dims)
    print(d.scales)
    print(d.data.shape)

    d = from_h5(voyager_h5)

    print(d)
    print(d.attrs)
    print(d.dims)
    print(d.scales)
    print(d.data.shape)

if __name__ == "__main__":
    test_load_voyager()