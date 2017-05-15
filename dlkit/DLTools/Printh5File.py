import h5py
from sys import argv

f = h5py.File(argv[1], "r")

print f.keys()

for k in f:
    print k
    try:
        print f[k].keys()
    except:
        print f[k].shape



# f.close()
