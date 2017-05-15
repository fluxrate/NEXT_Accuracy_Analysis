import time
import sys
import os
import signal
import subprocess as sp
from keras.utils import np_utils

import numpy as np
import h5py

from multiprocessing import Process, Pipe, Array, Value
from multiprocessing import Queue as PQueue
import ctypes

from threading import Thread
from Queue import Queue as TQueue
from Queue import Empty,Full
import glob

try:
    from subprocess import DEVNULL # py3k
except ImportError:
    import os
    DEVNULL = open(os.devnull, 'wb')

CHECKPARENTRATE = 1000

class DLMultiProcessGenerator(object):
    def __init__(self, batchsize=2048, skip=0, max=1e12, n_threads=20,
                 multiplier=1, waittime=0.005, timing=False,
                 preprocessfunction=False, postprocessfunction=False,
                 Wrap=True, verbose=False, sharedmemory=True,
                 qmultiple=1, catchsignals=False, GeneratorTimeout=600,
                 shapes=[], cachefile=False, SharedDataQueueSize=1):

        self.batchsize = batchsize
        self.skip = skip
        self.max = int(max)
        self.postprocessfunction = postprocessfunction
        self.preprocessfunction = preprocessfunction
        self.Wrap = Wrap
        self.verbose = verbose
        self.n_threads = n_threads
        self.readsize = batchsize * multiplier
        self.waittime = waittime
        self.timing = timing
        self.Done = Value('b', 0)

        self.GeneratorTimeout=GeneratorTimeout
        
        self.ReaderRun = Value('b', 0)
        self.PiperRun = Value('b', 0)
        self.FillerRun = Value('b', 0)
        
        self.qmultiple = qmultiple

        self.catchsignals = catchsignals

        self.workers = []
        self.count = 0
        self.Start = False
        self.Filler = False

        self.q_in = False
        self.q_Tout = False

        self.stopthreads = False
        self.preloaded = False

        self.shareddata = []
        self.DataQueueSize = SharedDataQueueSize
        self.shapes = shapes

        self.sig1 = signal.getsignal(signal.SIGINT)
        self.sig2 = signal.getsignal(signal.SIGTERM)

        self.parentPID = os.getpid()
        self.parentchecks = 0

        self.cachefilename=cachefile
            
        if sharedmemory:
            self.CreateSharedData()
            
        self.deletecachefile=False

    def __del__(self):
        if self.deletecachefile:
            print "Removing Cache File:",self.cachefilename
            os.remove(self.cachefilename)

        pass
        # try:
        #     self.kill_child_processes()
        # except:
        #     print "Failed to explicitly kill child processes."

    # Need to know the shapes to build the shared Tensors
    # But since reading is done if overloaded classes,
    # use them to read one set of data... using pipes
    # then reconfigure and use shared memory.
    def GetShapes(self):
        if self.verbose:
            print "In GetShapes."
        if len(self.shapes) > 0:
            return
        # Set workers to 1
        n_threads = self.n_threads
        self.n_threads = 1
        # set max to readsize
        max = self.max
        self.max = self.readsize
        Wrap = self.Wrap
        self.Wrap = False

        postprocessfunction = self.postprocessfunction
        self.postprocessfunction = False
        # use Generator to get examples until done.

        ExampleData = self.Generator().next()

        # Store shapes
        self.shapes = []
        for D in ExampleData:
            self.shapes.append(D.shape)

        # make sure everything is dead and reset
        self.StopFiller()
        self.StopWorkers()

        self.n_threads = n_threads
        self.max = max
        self.postprocessfunction = postprocessfunction
        self.Wrap = Wrap

        if not self.readsize == self.batchsize:
            for i in range(len(self.shapes)):
                if self.shapes[i][0] == self.batchsize:
                    self.shapes[i] = (self.readsize,) + self.shapes[i][1:]

        if self.verbose:
            print "Shapes:",self.shapes
            print "Done with GetShapes."

    def CreateSharedData(self):
        # GetShapes
        self.GetShapes()
        if self.verbose:
            print "GOT SHAPES:", self.shapes
        # Create data

        self.shareddata = []
        for i in range(self.n_threads):
            DsQ = []
            for j in range(self.DataQueueSize):
                Ds = []
                for D in self.shapes:
                    if self.verbose:
                        print "Creating Shared Data of Shape", D
                    shared_array_base = Array(ctypes.c_float, np.prod(D), lock=False)
                    shared_array = np.ctypeslib.as_array(shared_array_base)
                    shared_array = shared_array.reshape(D)
                    Ds.append(shared_array)
                State = Value('i', 0)
                DsQ.append([State, Ds])
            self.shareddata.append(DsQ)

    def SendData(self, D, i, p):
        if len(self.shareddata) > 0:
            FoundSlot = False
            waittime=self.waittime
            while not FoundSlot and self.check_parent(CHECKPARENTRATE):
                for k in range(self.DataQueueSize):
                    if self.shareddata[i][k][0].value == 0:
                        FoundSlot = True
                        waittime=self.waittime
                        break
                if not FoundSlot:
                    if waittime<1.:
                        waittime=2*waittime
                    if self.verbose:
                        print "SendData: waiting",waittime
                    time.sleep(waittime)
            for j, d in enumerate(D):
                self.shareddata[i][k][1][j][...] = d
            self.shareddata[i][k][0].value = 1
            if self.verbose:
                print "SendData", i, "sending", (i, k)

            p.send((i, k))
        else:
            p.send(D)

    def StartPQueues(self):
        self.q_in = PQueue(maxsize=self.qmultiple * self.n_threads)
        self.q_Tout = TQueue(maxsize=self.qmultiple * self.n_threads)

    def StopPQueues(self):
        if self.q_in:
            self.q_in.close()

        self.q_in = False
        self.q_Tout = False

    def StartWorkers(self):
        if len(self.workers) > 0:
            self.StopWorkers()

        if self.verbose:
            print "Starting Workers..."

        self.Done.value = False
        self.ReaderRun.value = True
        self.PiperRun.value = True
        
        self.workers = []
        self.Tworkers = []

        self.pipes = []
        for i in range(self.n_threads):
            if self.verbose:
                print "Starting Worker ", i

            parent_conn, child_conn = Pipe(True)
            self.pipes.append((parent_conn, child_conn))

            worker = Process(target=self.__DataReader, args=(i, child_conn,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

            Tworker = Thread(target=self.__DataPiper, args=(i, parent_conn,))
            Tworker.daemon = True
            Tworker.start()
            self.Tworkers.append(Tworker)

        self.Start = True

    def StartFiller(self):
        if self.Filler:
            print "Filler running... not restarting."
            return

        if self.verbose:
            print "Starting Filler..."

        self.FillerRun.value = True
        self.Filler = Process(target=self.__PQueueFiller, args=())
        self.Filler.daemon = True
        self.Filler.start()

    def StopWorkers(self):
        self.ReaderRun.value = False

        try:
            while not self.q_in.empty():
                self.q_in.get()
        except:
            pass
        
        for worker in self.workers:
            worker.terminate()
            # worker.join()

        self.workers = []

        try:
            while not self.q_Tout.empty():
                self.q_Tout.get()
        except:
            pass

        self.Tworkers = []

        self.Start = False

    def StopFiller(self):
        if not self.Filler:
            return

        self.FillerRun.value = True

        while not self.q_in.empty():
            self.q_in.get()

            #self.Filler.join()
        self.Filler.terminate()

        self.Filler = False

    def StartAll(self):
        self.StartPQueues()
        self.StartWorkers()
        self.StartFiller()

    def StopAll(self):
        try:
            self.StopFiller()
            self.StopWorkers()
            self.StopPQueues()
            self.kill_child_processes()
        except:
            pass

    def AllDone(self):  # Reset the Filler Process so we start at beginning at sample
        self.StopFiller()
        #self.Done.value = True
        if self.verbose:
            print "All Done."

    def postprocess(self, D):
        if self.postprocessfunction:
            return (self.postprocessfunction(D))
        else:
            return D

    def Generator(self):
        Done = False
        self.Done.value = False

        if self.verbose:
            print "Starting Generator."

        self.StartAll()

        while not Done:
            Done = not self.Wrap
            self.Done.value = False
            if self.verbose:
                print "Main Generator Loop."

            self.count = 0
            while self.Done.value==0:
                if self.verbose:
                    print "2nd Generator Loop."

                if not self.Filler:
                    self.StartFiller()

                if self.verbose or self.timing:
                    start = time.time()
                    print "Generator Waiting for data.", self.q_Tout.qsize()
                try:
                    GOT = self.q_Tout.get(True,self.GeneratorTimeout)
                except Empty:
                    print "Generator: No data for ", self.GeneratorTimeout, " seconds. Will attempt to restart."
                    self.StopAll()
                    self.StartAll()
                    if self.verbose:
                        print "Generator Done."
                    self.Done.value=True
                    Done=True
                    break
                    
                # print "GOT:",GOT
                if type(GOT) == tuple and len(GOT) == 2 and type(GOT[0]) == int and type(GOT[1]) == int:
                    ProcessI = GOT[0]
                    j = GOT[1]
                    D = self.shareddata[ProcessI][j][1]
                    if self.verbose or self.timing:
                        start1=time.time()
                    self.shareddata[ProcessI][j][0].value = 0
                    if self.verbose or self.timing:
                        print "Gen Set Done:",time.time()-start1
                else:
                    [ProcessI, D] = GOT

                if self.verbose or self.timing:
                    print "Generator got data from reader/piper.", ProcessI, " Fetch Time:", time.time() - start

                if self.readsize == self.batchsize:
                    if self.count + self.batchsize <= self.max:
                        self.count += self.batchsize
                        yield self.postprocess(D)
                    else:
                        out = []
                        remainder = self.max - self.count
                        if self.verbose:
                            "Generator remainder",remainder
                        for d in D:
                            out.append(d[:remainder])  # Ok to drop data here... 

                        if self.verbose:
                            print "Reached limit."
                            sys.stdout.flush()

                        self.count += remainder

                        self.AllDone()
                        if self.verbose:
                            print "Generator Done."
                            sys.stdout.flush()
                        self.Done.value=not self.Wrap
                        yield self.postprocess(out)
                        break
                else:
                    stoprun = False
                    if self.verbose:
                        print "Generator Readsize, Batchsize",self.readsize,self.batchsize
                    for i in xrange(0, self.readsize, self.batchsize):
                        out = []
                        if self.count + self.batchsize <= self.max:
                            for d in D:
                                out.append(d[i:i + self.batchsize])
                            self.count += self.batchsize
                            yield self.postprocess(out)
                        else:
                            remainder = self.max - self.count
                            if self.verbose:
                                print "Generator remainder",remainder
                            for d in D:
                                out.append(d[:remainder])

                            if self.verbose:
                                print "Reached limit."
                                sys.stdout.flush()

                            self.count += remainder
                            self.AllDone()
                            if self.verbose:
                                print "Generator Done."
                            self.Done.value=not self.Wrap
                            stoprun = True
                            yield self.postprocess(out)
                            break
                        if stoprun:
                            break

        self.StopAll()

        if self.verbose:
            print "Generator End."

    def check_parent(self, N=1):
        #return True
        try:
            self.parentchecks += 1
        except:
            return True

        if self.parentchecks % N == 0:
            parent_id = os.getppid()

            ps_command = sp.Popen("ps -o pid | grep %d" % parent_id, shell=True, stdout=sp.PIPE, stderr=DEVNULL)
            ps_output = ps_command.stdout.read()
            retcode = ps_command.wait()

            if len(ps_output) == 0 or parent_id == 1:
                print os.getpid(),"Dead parent!"
                os.kill(os.getpid(), signal.SIGKILL)
                return False
            else:
                # print ps_output
                return True
        else:
            return True

    def signal_handler(self, insignal, frame):
        # Check if parent is alive... if so... ignore. Else, die.
        if not self.check_parent():
            os.kill(os.getpid(), signal.SIGKILL)
        else:
            if self.verbose:
                print "Process:", os.getpid(), "ignoring kill."

    def ResetSignals(self):
        try:
            signal.signal(signal.SIGINT, self.sig1)
            signal.signal(signal.SIGTERM, self.sig2)
        except:
            pass

    def SetupSignals(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def DataPiper(self, i, p):
        Done = False
        while self.PiperRun.value==1 and not Done and self.check_parent(CHECKPARENTRATE):

            start = time.time()

            if self.verbose:
                print "Piper", i, " waiting for data."

            try:
                D = p.recv()
                # print D
            except:
                if self.verbose:
                    print "Piper Recieve Failed."
                D = False

            if self.verbose or self.timing:
                print i, "Piper Recieve Done in", time.time() - start

            if type(D) == bool:
                if self.verbose:
                    print i, "Piper Recieved Kill."
                Done = True
                break

            if self.verbose or self.timing:
                start = time.time()
                if self.verbose:
                    print i, " Piper waiting to send data.", self.q_Tout.qsize()

            if self.q_Tout:
                if self.verbose:
                    print i, " Piper sending data to generator."

                sent=False
                waittime=self.waittime
                while self.PiperRun.value==1 and not sent and not isinstance(self.q_Tout,bool):
                    try:
                        if type(D) == tuple and len(D) == 2 and type(D[0]) == int and type(D[1]) == int:
                            self.q_Tout.put(D,waittime)
                        else:
                            self.q_Tout.put([i, D],waittime)
                        self.q_Tout.task_done()
                    except Full:
                        if self.verbose:
                            print i,"Piper waiting for free slot.",waittime
                        if waittime<1:
                            waitime=waittime*2
                    except AttributeError:
                        send=True
            else:
                if self.verbose or self.timing:
                    print i, "Piper: No queue."
                break

            if self.verbose or self.timing:
                print i, "Piper Send Done in", time.time() - start

        if self.verbose:
            print "Exiting DataPiper", i

    def PreloadGenerator(self,n_threads=0):
        WrapRecall = self.Wrap
        postprocessRecall = self.postprocessfunction
        Done = False
        while not Done:
            if not self.preloaded:
                self.D = []
                first = True
                i = 0
                self.Wrap = False
                self.postprocessfunction = False
                if n_threads<1:
                    gen=self.Generator()
                else:
                    gen=self.DiskCacheGenerator(n_threads)
                for D in gen:
                    if first:
                        first = False
                        for j, T in enumerate(D):
                            T0 = np.zeros((self.max + self.batchsize,) + T.shape[1:])
                            T0[0:self.batchsize] = T
                            self.D.append(T0)

                    for j, T in enumerate(D):
                        try:
                            self.D[j][i:i + T.shape[0]] = T
                        except:
                            print "Something went wrong..."
                            print i, j, T.shape, D[j].shape

                    i += self.batchsize
                    if i>self.max:
                        break
                    if postprocessRecall:
                        yield postprocessRecall(D)
                    else:
                        yield D
                self.StopAll()
                self.preloaded = True
            else:
                for i in xrange(0, self.max, self.batchsize):
                    out = []
                    for d in self.D:
                        if i + self.batchsize >= self.max:
                            remainder = self.max - i
                            endi = i + remainder
                        else:
                            endi = i + self.batchsize
                        out.append(d[i:endi])

                    yield self.postprocess(tuple(out))

            self.Wrap = WrapRecall
            self.postprocessfunction = postprocessRecall
            Done = not self.Wrap

    def DiskCacheGenerator(self,n_threads=0):
        if n_threads<1:
            n_threads=self.n_threads
            
        WrapRecall = self.Wrap
        postprocessRecall = self.postprocessfunction
        Done = False

        renamecachefile=False
        if not self.preloaded:
            if self.cachefilename:
                self.deletecachefile=False
                if not os.path.exists(self.cachefilename):                    
                    renamecachefile=self.cachefilename.strip(".h5")+"-"+str(self.max)+".h5"
                else:
                    renamecachefile=self.cachefilename
                if os.path.exists(renamecachefile):
                    self.cachefilename=renamecachefile
                    self.preloaded=True
                else:
                    found=False
                    files=glob.glob(self.cachefilename.strip(".h5")+"*.h5")
                    for file in files:
                        try:
                            N_in_File=int(file.split("-")[-1].strip(".h5"))
                            if self.max<N_in_File:
                                found=file
                                break
                        except:
                            pass
                    if found:
                        self.cachefilename=found
                    else:
                        self.cachefilename=self.cachefilename.strip(".h5")+"-"+str(self.max)+"-PID"+str(os.getppid())+".h5"
            else:
                self.deletecachefile=True
                self.cachefilename="/tmp/"+os.environ["USER"]+"-"+str(os.getppid())+".h5"
        
        while not Done:
            if not self.preloaded:
                self.preloaded = True
                self.D = []
                first = True
                i = 0
                self.Wrap = False
                self.postprocessfunction = False
                for D in self.Generator():
                    if first:
                        first = False
                        f=h5py.File(self.cachefilename,"w")
                        for j, T in enumerate(D):
                            T0= f.create_dataset("dset"+str(j),(self.batchsize,) + T.shape[1:],
                                                 compression="lzf",
                                                 chunks=(self.batchsize,)+ T.shape[1:],
                                                 maxshape=(None,)+ T.shape[1:])
                            T0[0:self.batchsize] = T
                            self.D.append(T0)
                    else:
                        for j, T in enumerate(D):
                            #try:
                            self.D[j].resize( self.D[j].shape[0]+T.shape[0],axis=0)
                            self.D[j][i:i + T.shape[0]] = T
                            #except:
                            #    print "Something went wrong..."
                            #    print i, j, T.shape, D[j].shape

                    i += self.batchsize
                    if i>self.max:
                        break
                    if postprocessRecall:
                        yield postprocessRecall(D)
                    else:
                        yield D
                self.D=[]
                f.close()
                if renamecachefile:
                    if not os.path.exists(renamecachefile):
                        os.rename(self.cachefilename,renamecachefile)
                    else:
                        os.remove(self.cachefilename)
                    self.cachefilename=renamecachefile
                #self.StopAll()
            else:
                #self.StopAll()
                dsetnames=[]
                for j in xrange(len(h5py.File(self.cachefilename,"r").keys())):
                         dsetnames.append("dset"+str(j))

                gen=DLh5FileGenerator([self.cachefilename],dsetnames,
                                      max=self.max,
                                      n_threads=n_threads,
                                      batchsize=self.batchsize,
                                      postprocessfunction=postprocessRecall,
                                      verbose=self.verbose,
                                      timing=self.timing,
                                      shapes=self.shapes,
                                      Wrap=WrapRecall)
                Done = not self.Wrap

                for D in gen.Generator():
                    yield D

                gen.StopAll()
                
            self.Wrap = WrapRecall
            self.postprocessfunction = postprocessRecall
            Done = not self.Wrap
            
    def PreloadData(self,n_threads=0):
        #self.StopFiller()
        #self.StopWorkers()
        self.StopAll()
        WrapRecall = self.Wrap
        self.Wrap = False
        gen = self.PreloadGenerator(n_threads)
            
        if not self.preloaded:
            for D in gen:
                print ".",
                pass
            print
        self.Wrap = WrapRecall

    def CacheData(self,n_threads=4):
        #self.StopFiller()
        #self.StopWorkers()
        self.StopAll()
        WrapRecall = self.Wrap
        self.Wrap = False
        gen = self.DiskCacheGenerator(n_threads)
        if not self.preloaded:
            for D in gen:
                print ".",
                pass
            print
        self.Wrap = WrapRecall
        
    def kill_child_processes(self, signum=signal.SIGKILL):
        parent_id = self.parentPID  # os.getpid()
        ps_command = sp.Popen("pgrep -P %d" % parent_id, shell=True, stdout=sp.PIPE, stderr=DEVNULL)
        ps_output = ps_command.stdout.read()
        # TODO retcode is not used
        retcode = ps_command.wait()
        for pid_str in ps_output.strip().split("\n")[:-1]:
            try:
                os.kill(int(pid_str), signum)
            except:
                pass
    def __DataReader(self, i, p):
        if self.catchsignals:
            self.SetupSignals()
        self.DataReader(i, p)

        if self.catchsignals:
            self.ResetSignals()

    def __DataPiper(self, i, p):
        self.DataPiper(i, p)


    def __PQueueFiller(self):
        if self.catchsignals:
            self.SetupSignals()
        self.PQueueFiller()

        if self.catchsignals:
            self.ResetSignals()


class DLh5FileGenerator(DLMultiProcessGenerator):
    def __init__(self, files, datasets, **kwargs):
        self.files = files
        self.datasets = datasets
        super(DLh5FileGenerator, self).__init__(**kwargs)

    def DataReader(self, i, p):
        currentfileI = False
        F = False
        Done=False

        while not Done and self.ReaderRun.value==1 and self.check_parent(CHECKPARENTRATE):
            [fileI, index] = self.q_in.get()

            if fileI == -1:
                if self.verbose:
                    print i, "Reader Got kill signal."
                    sys.stdout.flush()
                Done = True
                break

            if not F or fileI != currentfileI:
                if F:
                    F.close()
                F = h5py.File(self.files[fileI], "r")
                currentfileI = fileI

            Ins = []

            for D in self.datasets:
                Ins.append(F[D][index:index + self.readsize])

            if self.preprocessfunction:
                Ins = self.preprocessfunction(Ins)

            if self.verbose or self.timing:
                print i, "Reader Sending through pipe."
                start = time.time()

            self.SendData(Ins, i, p)
            if self.verbose or self.timing:
                print i, "Reader Done in ", time.time() - start

        if F:
            F.close()

        p.send(False)

        if self.verbose:
            print i, "Exiting DataReader Process."
            sys.stdout.flush()

        time.sleep(1)

    def PQueueFiller(self):
        if self.verbose:
            print "Filler Loop start."

        while self.FillerRun.value==1 and self.check_parent(CHECKPARENTRATE):
            for fileI, file in enumerate(self.files):
                index = self.skip

                if self.verbose:
                    print "Filler: Openning file:",file
                F = h5py.File(file, "r")
                
                X_In_Shape = F[self.datasets[0]].shape
                N = X_In_Shape[0]
                F.close()

                if self.verbose:
                    print index,self.readsize, N
                while index + self.readsize < N and self.FillerRun.value==1 and self.check_parent(
                        CHECKPARENTRATE):  # This potentially skips the last events in the file.
                    stop = False
                    if self.timing or self.verbose:
                        print "Filler sending data."
                        start = time.time()
                    while not stop and self.FillerRun.value==1 and self.check_parent(CHECKPARENTRATE):
                        if not self.q_in.full():
                            self.q_in.put([fileI, index])
                            index += self.readsize
                            stop = True
                            if self.timing or self.verbose:
                                print "Filler send data in", time.time() - start
                        else:
                            # if self.timing or self.verbose:
                            #    print "Waiting:",self.waittime
                            time.sleep(self.waittime)


        if self.verbose:
            print "Exiting PQueueFiller Process."
            sys.stdout.flush()

class DLMultiClassGenerator(DLMultiProcessGenerator):
    def __init__(self, Samples, batchsize=2048,
                 OneHot=True, ClassIndex=False, ClassIndexMap=False,
                 closefiles=False,
                 sleep=0,**kargs):
        self.Samples = Samples
        self.closefiles=closefiles
        self.Classes = self.OrganizeFiles(ClassIndexMap=ClassIndexMap)
        self.OneHot = OneHot  # Return one hot?
        self.ClassIndex = ClassIndex  # Return class index
        self.ClassIndexMap = ClassIndexMap  # The ClassIndex Map. Use external one if got one.
        self.sleep=sleep

        self.ClassNames = self.Classes.keys()

        # Store a map between class names and index.
        if not self.ClassIndexMap:
            self.ClassIndexMap = {}
            for C in self.Classes:
                self.ClassIndexMap[C] = self.Classes[C]["ClassIndex"]

        self.N_Classes = len(self.Classes.keys())
        self.N_ExamplePerClass = int(batchsize / self.N_Classes)
        self.remainEx = batchsize - self.N_Classes * self.N_ExamplePerClass

        super(DLMultiClassGenerator, self).__init__(batchsize=batchsize, **kargs)

        if self.verbose:  # Does this assume same number of events in every file?
            print "Found", self.N_Classes, " classes. Will pull", self.N_ExamplePerClass, " examples from each class."
            print "Will have", self.remainEx, "remaining.. will randomly pad classes."

    def PQueueFiller(self):
        if self.verbose:
            print "Filler Loop start."

        while self.FillerRun.value==1 and self.check_parent(CHECKPARENTRATE):
            # Assign how many events per class we will use

            for C in self.Classes:  # Should this be Possion?
                self.Classes[C]["NExamples"] = self.N_ExamplePerClass

            # Add events randomly from classes to make the batch full.
            for i in xrange(0, int(self.remainEx)):
                ii = int(float(self.N_Classes) * np.random.random())
                self.Classes[self.ClassNames[ii]]["NExamples"] += 1

            count = 0
            N_TotalExamples = 0

            BatchData = []
            for C in self.Classes:
                if self.Done.value:
                    break
                count += 1
                Cl = self.Classes[C]

                # Look over the files in this class
                N_Examples = 0
                while N_Examples < Cl["NExamples"] and not self.Done.value and self.check_parent(CHECKPARENTRATE):
                    if Cl["File_I"] >= len(Cl["Files"]):
                        print "Warning: out of files for", C  # ,N_Examples,self.Done.value,CL["File_I"]
                        if not self.Wrap:
                            if self.verbose:
                                print "Stopping Filler."
                            self.Done.value = True
                            Cl["File_I"] = 0
                            time.sleep(1)
                            break
                        else:
                            if self.verbose:
                                print "Wrapping. Starting with first file for", C
                            Cl["File_I"] = 0

                    if self.verbose:
                        print count, "/", self.N_Classes, ":", C, ":", Cl["Files"][Cl["File_I"]]

                    if Cl["Example_I"] == 0:
                        if "File" in Cl and Cl["File"]:
                            Cl["File"].close()
                        if self.verbose:
                            print "Opening:", C, Cl["File_I"], Cl["Files"][Cl["File_I"]]
                        if self.timing:
                            ostart=time.time()
                        f = Cl["File"] = h5py.File(Cl["Files"][Cl["File_I"]], "r")
                        if self.timing:
                            print "Filler: File Open Time: ",time.time()-ostart
                            
                        N = Cl["N_File"][Cl["File_I"]] = f[Cl["DataSetName"][0]].shape[0]
                        f.close()
                        if self.verbose:
                            print " with", N, " events."
                        Cl["File"] = False

                    N = Cl["N_File"][Cl["File_I"]]
                    I = Cl["Example_I"]
                    N_Unused = N - I
                    N_End = min(I + (Cl["NExamples"] - N_Examples), N)
                    N_Using = N_End - I

                    if self.verbose:
                        print "N,I,N_Unused,N_End,N_Using", N, I, N_Unused, N_End, N_Using

                    BatchData.append([C, Cl["File_I"], I, N_End])

                    N_Examples += N_Using
                    N_TotalExamples += N_Using
                    if N_End >= N:
                        Cl["Example_I"] = 0
                        Cl["File_I"] += 1
                    else:
                        Cl["Example_I"] = N_End

            # Now Send the BatchData to the workers
            Stop = False
            while not Stop and self.FillerRun.value==1 and self.check_parent(CHECKPARENTRATE):
                if not self.q_in.full():
                    if self.verbose:
                        print "Filler sending request", BatchData
                    self.q_in.put(BatchData)
                    Stop = True
                else:
                    time.sleep(self.waittime)

        if self.verbose:
            print "Filler exiting."

    def DataReader(self, i, p):
        firstIndex = True

        if self.sleep>0:
            time.sleep(self.sleep*i)

        Data = []
        while self.ReaderRun.value==1 and self.check_parent(CHECKPARENTRATE):
            BatchData = self.q_in.get()
            if self.verbose:
                print i, "Data Reader got request: ", BatchData

            # Check if we got the kill signal
            if len(BatchData[0]) == 1:
                if self.verbose:
                    print i, "Request < 3 len... ", BatchData
                break

            N_TotalExamples = 0
            # Loop over file/chunck requests for this batch
            for BD in BatchData:
                if self.verbose:
                    print i, "Data Reader processing request: ", BD
                C = BD[0]
                FileI = BD[1]
                Cl = self.Classes[C]
                I = BD[2]
                N_End = BD[3]
                N_Using = N_End - I

                # Open the file.
                if not Cl["File"] or FileI != Cl["File_I"]:
                    if Cl["File"]:
                        Cl["File"].close()
                    if self.verbose:
                        print i, "Data Reader Opening:", Cl["File_I"]
                    Cl["File_I"] = FileI
                    Cl["File"] = h5py.File(Cl["Files"][FileI], "r")
                f = Cl["File"]
                # Fill class index based on samples definition.
                if self.ClassIndex or self.OneHot:
                    if firstIndex:
                        IndexT = np.zeros(self.readsize)
                        firstIndex = False
                    # Fill the index
                    a = np.empty(N_Using);
                    a.fill(Cl["ClassIndex"])
                    IndexT[N_TotalExamples:N_TotalExamples + N_Using] = a
                    if self.verbose:
                        print i, "Data Reader Cat'ing index from ", C, Cl["ClassIndex"]

                # Fill the data from file
                for ii, DataSetName in enumerate(Cl["DataSetName"]):
                    finalShape = f[DataSetName].shape
                    finalShape = (self.readsize,) + finalShape[1:]

                    if len(Data) < ii + 1:
                        Data.append(np.zeros(finalShape,dtype=f[DataSetName].dtype))
                    if self.verbose:
                        print i, "Data Reader Cat'ing data from ", C, DataSetName, ii, "out: ",
                        print N_TotalExamples, N_TotalExamples + N_Using, " in:", I, N_End, f[DataSetName].shape
                        sys.stdout.flush()
                    if f[DataSetName].shape[0] < N_End:
                        print i, "Error: Data Reader mismatch", Cl["File_I"], Cl["Files"][Cl["File_I"]], \
                            f[DataSetName].shape[0], N_End
                        sys.stdout.flush()
                    
                    Data[ii][N_TotalExamples:N_TotalExamples + N_Using] = f[DataSetName][I:N_End]

                if self.closefiles:
                    Cl["File"].close()
                    Cl["File"]=False

                N_TotalExamples += N_Using

            out = tuple(Data)

            if self.ClassIndex:
                out += (IndexT,)

            if self.OneHot:
                Y1 = np_utils.to_categorical(IndexT, self.N_Classes)
                out += (Y1,)

            out = self.shuffle_in_unison_inplace(out)

            # print out[-1]
            # print out[0][np.where(out[0]!=0.)]

            if self.verbose:
                print i, "Data Reader sending", len(out), " objects."

            if self.preprocessfunction:
                out = self.preprocessfunction(out)

            self.SendData(out, i, p)

        if self.verbose:
            print i, "Exiting DataReader Process."
            sys.stdout.flush()

        # Tell the correspinding thread to stop.
        p.send(False)

        # Close all files
        for C in self.Classes:
            try:
                self.Classes[C]["File"].close()
            except:
                pass

        time.sleep(1)

    def OrganizeFiles(self, OpenFiles=False, ClassIndexMap=False):
        Files = {}

        NFiles = len(self.Samples)
        index = 0
        for S in self.Samples:
            if len(S) == 2:
                ClassName = DataSetName = S[1]
                File = S[0]

            if len(S) == 3:
                DataSetName = S[1]
                ClassName = S[2]
                File = S[0]

            if not ClassName in Files.keys():
                if ClassIndexMap:
                    index = ClassIndexMap[ClassName]

                Files[ClassName] = {"N": 0,
                                    "Files": [],
                                    "N_File": [],
                                    "File_I": 0,
                                    "Example_I": 0,
                                    "DataSetName": DataSetName,
                                    "ClassIndex": index,
                                    "File": False}
                if not ClassIndexMap:
                    index += 1

            if OpenFiles:
                print "Opening", index, "/", NFiles, ":", S
                sys.stdout.flush()
                try:
                    f = h5py.File(File)
                except:
                    print
                    print "Failed Opening:", S
                    continue

                print DataSetName
                N = np.shape(f[DataSetName])[0]
                Files[ClassName]["N"] += N
                f.close()

            Files[ClassName]["Files"].append(File)
            Files[ClassName]["N_File"].append(-1)

        self.ClassIndexMap=ClassIndexMap
        return Files

    def shuffle_in_unison_inplace(self, Data):
        N = len(Data[0])
        p = np.random.permutation(N)

        out = []
        for d in Data:
            assert N == len(d)
            out.append(d[p])

        return out


class DLMultiClassFilterGenerator(DLMultiClassGenerator):
    def __init__(self, Samples, FilterFunc, **kargs):
        self.FilterFunc=FilterFunc
        super(DLMultiClassFilterGenerator, self).__init__(Samples, **kargs)

    def PQueueFiller(self):
        if self.verbose:
            print "Filler Loop start."

        while self.FillerRun.value==1 and self.check_parent(CHECKPARENTRATE):
            # Assign how many events per class we will use

            for C in self.Classes:  # Should this be Possion?
                self.Classes[C]["NExamples"] = self.N_ExamplePerClass

            # Add events randomly from classes to make the batch full.
            for i in xrange(0, int(self.remainEx)):
                ii = int(float(self.N_Classes) * np.random.random())
                self.Classes[self.ClassNames[ii]]["NExamples"] += 1

            count = 0
            N_TotalExamples = 0
            BatchData = []
            for C in self.Classes:
                if self.Done.value:
                    break
                count += 1
                Cl = self.Classes[C]

                # Look over the files in this class
                N_Examples = 0
                while N_Examples < Cl["NExamples"] and not self.Done.value and self.check_parent(CHECKPARENTRATE):
                    if Cl["File_I"] >= len(Cl["Files"]):
                        print "Warning: out of files for", C  # ,N_Examples,self.Done.value,CL["File_I"]
                        if not self.Wrap:
                            if self.verbose:
                                print "Stopping Filler."
                            self.Done.value = True
                            Cl["File_I"] = 0
                            time.sleep(1)
                            break
                        else:
                            if self.verbose:
                                print "Wrapping. Starting with first file for", C
                            Cl["File_I"] = 0

                    if self.verbose:
                        print count, "/", self.N_Classes, ":", C, ":", Cl["Files"][Cl["File_I"]]

                    if Cl["Example_I"] == 0:
                        if "File" in Cl and Cl["File"]:
                            Cl["File"].close()
                        if self.verbose:
                            print "Opening:", C, Cl["File_I"], Cl["Files"][Cl["File_I"]]
                        f = Cl["File"] = h5py.File(Cl["Files"][Cl["File_I"]], "r")
                        if self.FilterFunc:
                            Cl["Index"]=self.FilterFunc(f)
                        else:
                            Cl["Index"]=range(f[Cl["DataSetName"][0]].shape[0])
                        N = Cl["N_File"][Cl["File_I"]] = len(Cl["Index"])
                        f.close()
                        if self.verbose:
                            print " with", N, " filtered events."
                            print "INDEX:",Cl["Index"]
                        Cl["File"] = False

                    N = Cl["N_File"][Cl["File_I"]]
                    I = Cl["Example_I"]
                    N_Unused = N - I
                    N_End = min(I + (Cl["NExamples"] - N_Examples), N)
                    N_Using = N_End - I

                    if self.verbose:
                        print "N,I,N_Unused,N_End,N_Using", N, I, N_Unused, N_End, N_Using

                    BatchData.append([C, Cl["File_I"], Cl["Index"][I:N_End]])

                    N_Examples += N_Using
                    N_TotalExamples += N_Using
                    if N_End >= N:
                        Cl["Example_I"] = 0
                        Cl["File_I"] += 1
                    else:
                        Cl["Example_I"] = N_End

            # Now Send the BatchData to the workers
            Stop = False
            while not Stop and self.FillerRun.value==1 and self.check_parent(CHECKPARENTRATE):
                if not self.q_in.full():
                    if self.verbose:
                        print "Filler sending request", BatchData
                    self.q_in.put(BatchData)
                    Stop = True
                else:
                    time.sleep(self.waittime)

        if self.verbose:
            print "Filler exiting."
    
    def DataReader(self, i, p):
        firstIndex = True

        if self.sleep>0:
            time.sleep(self.sleep*i)
        
        Data = []
        while self.ReaderRun.value==1 and self.check_parent(CHECKPARENTRATE):
            if self.timing:
                start=time.time()
            BatchData = self.q_in.get()
            if self.timing:
                print "DataReader: Waiting for instruction time",time.time()-start

            if self.verbose:
                print i, "DataReader got request: ", BatchData

            # Check if we got the kill signal
            if len(BatchData[0]) == 1:
                if self.verbose:
                    print i, "Request < 3 len... ", BatchData
                break

            N_TotalExamples = 0
            if self.timing:
                start=time.time()

            # Loop over file/chunck requests for this batch
            for BD in BatchData:
                if self.verbose:
                    print i, "DataReader processing request: ", BD
                C = BD[0]
                FileI = BD[1]
                Cl = self.Classes[C]
                I = BD[2]
                N_End= N_Using = len(I)
                
                # Open the file.
                if not Cl["File"] or FileI != Cl["File_I"]:
                    if Cl["File"]:
                        Cl["File"].close()
                    if self.verbose:
                        print i, "DataReader Opening:", Cl["File_I"]
                    Cl["File_I"] = FileI
                    Cl["File"] = h5py.File(Cl["Files"][FileI], "r")
                f = Cl["File"]
                # Fill class index based on samples definition.
                if self.ClassIndex or self.OneHot:
                    if firstIndex:
                        IndexT = np.zeros(self.readsize)
                        firstIndex = False
                    # Fill the index
                    a = np.empty(N_Using);
                    a.fill(Cl["ClassIndex"])
                    IndexT[N_TotalExamples:N_TotalExamples + N_Using] = a
                    if self.verbose:
                        print i, "DataReader Cat'ing index from ", C, Cl["ClassIndex"]

                # Fill the data from file
                for ii, DataSetName in enumerate(Cl["DataSetName"]):
                    finalShape = f[DataSetName].shape
                    finalShape = (self.readsize,) + finalShape[1:]

                    if len(Data) < ii + 1:
                        Data.append(np.zeros(finalShape))
                    if self.verbose:
                        print i, "DataReader Cat'ing data from ", C, DataSetName, ii, "out: ",
                        print N_TotalExamples, N_TotalExamples + N_Using, " in:", I, N_End, f[DataSetName].shape
                        sys.stdout.flush()
                    if f[DataSetName].shape[0] < N_End:
                        print i, "DataReader Error: Data Reader mismatch", Cl["File_I"], Cl["Files"][Cl["File_I"]], \
                            f[DataSetName].shape[0], N_End
                        sys.stdout.flush()
                    if len(I)>0:
                        Data[ii][N_TotalExamples:N_TotalExamples + N_Using] = f[DataSetName][np.array(I)-np.min(I),...]

                    if self.closefiles:
                        Cl["File"].close()
                        Cl["File"]=False

                N_TotalExamples += N_Using

            out = tuple(Data)

            if self.ClassIndex:
                out += (IndexT,)

            if self.OneHot:
                Y1 = np_utils.to_categorical(IndexT, self.N_Classes)
                out += (Y1,)

            out = self.shuffle_in_unison_inplace(out)

            if self.timing:
                print "DataReader: time to read data:",time.time()-start

            if self.verbose:
                print i, "DataReader sending", len(out), " objects."

            if self.timing:
                start=time.time()
                
            if self.preprocessfunction:
                out = self.preprocessfunction(out)

            if self.timing:
                print "DataReader: time to preprocess:",time.time()-start

            if self.timing:
                start=time.time()

            self.SendData(out, i, p)

            if self.timing:
                print "DataReader: time to send data:",time.time()-start

        if self.verbose:
            print i, "Exiting DataReader Process."
            sys.stdout.flush()

        # Tell the correspinding thread to stop.
        p.send(False)

        # Close all files
        for C in self.Classes:
            try:
                self.Classes[C]["File"].close()
            except:
                pass

        time.sleep(1)
        
    
Test = 1
if __name__ == '__main__' and Test == 0:
    InputFile = "/Users/afarbin/LCD/Data/LCD-Merged-All.h5"
    batchsize = 1024

    try:
        n_threads = int(sys.argv[1])
    except:
        n_threads = 4

    Train_gen = DLh5FileGenerator(files=[InputFile], datasets=["ECAL", "OneHot"],
                                  batchsize=batchsize, Wrap=False, multiplier=2,
                                  max=100000, verbose=False, timing=False, n_threads=n_threads)

    N = 1
    start = time.time()
    for tries in xrange(2):
        print "*********************Try:", tries

        for D in Train_gen.Generator():
            Delta = (time.time() - start)
            print Delta, ":", Delta / float(N)
            sys.stdout.flush()
            N += 1
            for jj, d in enumerate(D):
                print jj, d.shape
                # print d[np.where(d!=0)]
                pass

    print "Final time per batch:", Delta / float(N)

    
if __name__ == '__main__' and Test == 1:
    from CaloDNN.LCDData import LCDDataGenerator

    # h5keys=["ECAL","HCAL","target"]
    h5keys = ["ECAL"]
    NEvents = 1e8
    batchsize = 1024
    verbose = False
    MaxFiles = -1
    # Create a generator
    try:
        n_threads = int(sys.argv[1])
    except:
        n_threads = 4

    print "Using ", n_threads, "Processes/Threads."

    [Train_gen, IndexMap] = LCDDataGenerator(h5keys, batchsize, FileSearch="/Users/afarbin/LCD/Data/*/*.h5",
                                             verbose=verbose,
                                             OneHot=True, ClassIndex=False, ClassIndexMap=True,
                                             MaxFiles=MaxFiles,
                                             multiplier=2,
                                             n_threads=n_threads,
                                             timing=False)

    print IndexMap
    N = 1
    count = 0
    start = time.time()
    for tries in xrange(2):
        print "*********************Try:", tries
        for D in Train_gen.Generator():
            Delta = (time.time() - start)
            print count, ":", Delta, ":", Delta / float(N)
            sys.stdout.flush()
            N += 1
            for d in D:
                print d.shape
                NN = d.shape[0]
                # print d[0]
                pass
            count += NN
