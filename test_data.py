from hyperseti.data import from_fil, from_h5

filpath = '/home/dancpr/blimpy/tests/test_data/Voyager1.single_coarse.fine_res.fil'
h5path = '/home/dancpr/blimpy/tests/test_data/Voyager1.single_coarse.fine_res.h5'

d = from_fil(filpath)

print(d)
print(d.attrs)
print(d.dims)
print(d.scales)
print(d.data.shape)

d = from_h5(h5path)

print(d)
print(d.attrs)
print(d.dims)
print(d.scales)
print(d.data.shape)