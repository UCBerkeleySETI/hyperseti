from hyperseti.data import from_fil, from_h5
from file_defs import *

d = from_fil(VOYAFIL)

print(d)
print(d.attrs)
print(d.dims)
print(d.scales)
print(d.data.shape)

d = from_h5(VOYAH5)

print(d)
print(d.attrs)
print(d.dims)
print(d.scales)
print(d.data.shape)
