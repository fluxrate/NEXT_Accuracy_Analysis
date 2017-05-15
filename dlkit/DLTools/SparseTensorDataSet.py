#from tables import *
from scipy.sparse import find
import numpy as np
import h5py
# import sys

import threading


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g

class SparseTensorDataSet:
    def __init__(self, shape=(), default=0., unbinned=False, dtype="float32", Files=False):
        self.unbinned = unbinned
        self.shape = tuple(shape)
        self.__default = default  # default value of non-assigned elements
        self.ndim = len(shape)
        self.dtype = dtype
        self.C = []  # This will hold the sparse ND arrays
        self.V = []  # This will hold the sparse ND arrays

        self.Files = Files

    def shuffle(self):
        def shuffle_in_unison_inplace(a, b):
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            try:
                return a[p], b[p]
            except:
                return map(lambda i: a[i], p), map(lambda i: b[i], p),

        self.C, self.V = shuffle_in_unison_inplace(self.C, self.V)

    def append(self, Coordinates, Values):
        Cs = []
        Vs = []

        for C, V in zip(Coordinates, Values):
            Cs.append(tuple(C))
            Vs.append(V)

    def convertToBinned(self):
        ## Not completed... need to flatten, use find, and unflatten, and store
        ## in self.C,self.V. Requires reindexing.
        N = self.len()

        for i in xrange(B):
            out, binedges = self.histogram(i, bins)
            find(out)

        self.binedges = binedges

    def histogram(self, i, bins=False):
        if not (isinstance(bins, list) or isinstance(bins, tuple)):
            bins = self.shape
        # returns histogram and bin edges
        return np.histogramdd(self.C[i], bins=list(bins), weights=self.V[i])

    def histogramAll(self, range=False, bins=False):
        if not (isinstance(bins, list) or isinstance(bins, tuple)):
            bins = self.shape
        if range:
            N = range[1] - range[0]
        else:
            N = self.len()
        out = np.zeros((N,) + tuple(bins))

        if not range:
            range = xrange(N)
        else:
            range = xrange(range[0], range[1])

        for i in range:
            out[i], binedges = self.histogram(i, bins)

        return out, binedges

    # This only makes sense if the coordinates are integers
    def dense(self, i):
        if self.unbinned:
            return self.histogram(i)[0]

        a = np.zeros((1,) + self.shape, dtype=self.dtype)

        for C, V in zip(self.C[i], self.V[i]):
            a[tuple(C)] = V

        return a

    def denseAll(self, range=[]):
        if self.unbinned:
            return self.histogramAll(range)[0]

        if len(range) > 0:
            Start = range[0]
            Stop = range[1]
        else:
            Start = 0
            try:
                Stop = len(self.C)
            except:
                Stop = self.C.shape[0]

        a = np.zeros((Stop - Start,) + tuple(self.shape), dtype=self.dtype)

        for i in xrange(Start, Stop):
            for C, V in zip(self.C[i], self.V[i]):
                try:
                    a[i - Start][tuple(C)] = V
                except:
                    print "Reached End of Sample."

        return a

    def Writeh5(self, h5file, name, range=[]):
        root = h5file.create_group("/", name, name)
        FILTERS = Filters(complib='zlib', complevel=5)

        if self.unbinned:
            CT = h5file.create_vlarray(root, "C", Float32Atom(shape=len(self.shape)), "", filters=FILTERS)
        else:
            CT = h5file.create_vlarray(root, "C", Int32Atom(shape=len(self.shape)), "", filters=FILTERS)
        VT = h5file.create_vlarray(root, "V", Float32Atom(), "", filters=FILTERS)
        h5file.create_array(root, "shape", self.shape)
        h5file.create_array(root, "unbinned", [int(self.unbinned)])

        if len(range) > 0:
            Start = range[0]
            Stop = range[1]
        else:
            Start = 0
            Stop = len(self.C)

        for i in xrange(Start, Stop):
            CT.append(self.C[i])
            VT.append(self.V[i])

    def Readh5(self, f, name, range=[]):
        self.shape = np.array(f[name]["shape"])
        self.unbinned = bool(f[name]["unbinned"][0])
        if len(range) > 0:
            Start = range[0]
            Stop = range[1]
            self.C = f[name]["C"][Start:Stop]
            self.V = f[name]["V"][Start:Stop]

        else:
            self.C = f[name]["C"]
            self.V = f[name]["V"]

    def Readh5Files(self, filelist, name):
        for filename in filelist:
            f = h5py.File(filename, "r")
            self.shape = np.array(f[name]["shape"])

            try:
                self.C = np.concatenate(self.C, f[name]["C"])
                #                self.V=np.concatenate(self.V,f[name]["V"])
                self.V += list(self.V, f[name]["V"])
            except:
                self.C = np.array(f[name]["C"])
                self.V = list(f[name]["V"])
            f.close()

    def Readh5Files(self, filelist, name):  #### BROKEN!
        for filename in filelist:
            print "Reading :", filename,
            sys.stderr.flush()
            f = h5py.File(filename, "r")
            self.shape = np.array(f[name]["shape"])

            try:
                self.C = f[name]["C"]
                #                self.V=np.concatenate(self.V,f[name]["V"])
                self.V += f[name]["V"]
            except:
                self.C = np.array(f[name]["C"])
                self.V = list(f[name]["V"])
            f.close()

            print "Done."
            sys.stderr.flush()

    def DenseGenerator(self, BatchSize, Wrap=True):
        pass

    def FromDense(self, A, Clear=False):
        if Clear:
            self.C = []
            self.V = []
        for a in A:
            # X,Y,V=find(a)
            CC = np.where(a != 0.)
            V = a[CC]
            C = np.array(CC)
            C = C.transpose()

            self.C.append(tuple(C))
            self.V.append(V)

    def AppendFromDense(self, a):

        X, Y, V = find(a)
        C = np.array([X, Y])
        C = C.transpose()

        self.C.append(tuple(C))
        self.V.append(V)

    def len(self):
        try:
            N = len(self.C)
        except:
            N = self.C.shape[0]
        return N

    @threadsafe_generator
    def DenseGeneratorOld(self, BatchSize, Wrap=True):
        Done = False
        N = self.len()
        while not Done:
            for i in xrange(0, N - BatchSize, BatchSize):  # May miss some Examples at end of file... need better logic
                yield self.denseAll([i, i + BatchSize])
                Done = not Wrap


Test = 1
if __name__ == '__main__' and Test == 0:
    # Main
    import h5py
    import time

    shape = (10000, 10, 40, 9)
    density = 0.1
    batchsize = 100

    N_Examples = shape[0]
    N_Vals = np.prod(shape)

    print "Testing with tensor size", shape, " and density", density, "."

    # Generate Some Sparse Data
    start = time.time()
    Vals = np.array(np.random.random(int(N_Vals * density)), dtype="float32")
    Zs = np.zeros(N_Vals - Vals.shape[0], dtype="float32")
    Train_X = np.concatenate((Vals, Zs))
    np.random.shuffle(Train_X)
    Train_X = Train_X.reshape(shape)
    print "Time Generate Sparse Data (into a Dense Tensor):", time.time() - start

    # Now Create the sparse "Tensor"
    X = SparseTensorDataSet(Train_X.shape[1:])

    # Test making the data sparse. (Only works for 2d right now)
    start = time.time()
    X.FromDense(Train_X)
    print "Time to convert to Sparse:", time.time() - start

    print "Sparsity Achieved: ", float(sum(map(len, X.C))) / N_Vals

    # Write to File
    start = time.time()
    f = open_file("TestOut.h5", "w")
    X.Writeh5(f, "Data")
    f.close()
    print "Time to write out:", time.time() - start

    # Read back
    start = time.time()
    XX = SparseTensorDataSet(Train_X.shape[1:])
    f = h5py.File("TestOut.h5", "r")
    XX.Readh5(f, "Data")
    #    f.close()
    print "Time to read back:", time.time() - start

    # Try to reconstruct the original data
    start = time.time()
    XXX = XX.denseAll()
    print "Time to convert to Dense:", time.time() - start

    # Test
    print
    NValues = np.prod(Train_X.shape)
    T = Train_X == XXX
    NGood = np.sum(T)
    print "Number of non-matching values", NValues - NGood, "/", NValues

    print "The Difference:"
    print XXX[np.where(T == False)] - Train_X[np.where(T == False)]
    print "Average difference of mismatch terms:", np.sum(XXX[np.where(T == False)] - Train_X[np.where(T == False)]) / (NValues - NGood)

    start = time.time()
    print "Generator batchsize:", batchsize
    for D in XX.DenseGenerator(batchsize, False):
        print ".",

    print
    print "Time to run generator:", time.time() - start

if __name__ == '__main__' and Test == 1:
    from ThreadedGenerator import DLh5FileGenerator
    import glob, time, sys

    batchsize = 1024

    try:
        n_threads = int(sys.argv[1])
    except:
        n_threads = 4

    InputDirectory = "/data/wghilliard/out3/*/*.h5"
    InputFiles = glob.glob(InputDirectory)

    f = h5py.File(InputFiles[1], "r")
    TheShape = tuple(f["images/shape"])
    print TheShape
    f.close()


    def Denser(Ins):
        C = Ins[0]
        V = Ins[1]

        SparseT = SparseTensorDataSet(shape=TheShape)
        SparseT.C = C
        SparseT.V = V

        return [SparseT.denseAll()]


    Train_gen = DLh5FileGenerator(files=InputFiles, datasets=["images/C", "images/V"],
                                  preprocessfunction=Denser,
                                  batchsize=batchsize, Wrap=False,
                                  max=10000000, verbose=False, timing=False, n_threads=n_threads)

    N = 1
    start = time.time()
    for tries in xrange(2):
        print "*********************Try:", tries

        for D in Train_gen.Generator():
            Delta = (time.time() - start)
            print N, ":", Delta, ":", Delta / float(N)
            sys.stdout.flush()
            N += 1
            for d in D:
                print d.shape
                # print d[np.where(d!=0.)]
                # print d[0]
                pass
