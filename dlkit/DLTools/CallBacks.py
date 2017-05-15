import keras
from time import time
from sklearn.metrics import roc_curve, auc
import signal
import sys
import os
import subprocess as sp


class TimeStopping(keras.callbacks.Callback):
    def __init__(self, RunningTime, IgnoreFirstEpochs=1, StartTime=None, verbose=False):
        super(TimeStopping, self).__init__()
        self.history = []
        self.verbose = verbose
        self.IgnoreFirstEpochs=IgnoreFirstEpochs
        self.epoch=0
        
        self.RunningTime = RunningTime
        if StartTime:
            self.StartTime = StartTime
        else:
            self.StartTime = time()

        self.EpochStartTime = time()
        self.EpochEndTime = time()

        if self.verbose:
            print "TimeStopping Callback: RunTime set to", self.RunningTime, "."

    def on_epoch_start(self, logs={}):
        if self.verbose:
            print "TimeStopping Callback: Epoch Start."
        self.EpochStartTime = time()

    def on_epoch_end(self, batch, logs={}):
        if self.IgnoreFirstEpochs>self.epoch:
            self.EpochStartTime = time()
            self.epoch+=1
            return
        self.epoch+=1
        self.EpochEndTime = time()
        self.EpochTime = self.EpochEndTime - self.EpochStartTime
        self.history.append(self.EpochTime)
        if self.verbose:
            print
            print "TimeStopping Callback: EpochTime is", self.EpochTime, ". Running Time is", time() - self.StartTime, " / ", self.RunningTime, "."
        if (time() - self.StartTime + self.EpochTime) > self.RunningTime:
            print
            print "TimeStopping Callback: Out of time. Stopping Training."
            print "EpochTime=", self.EpochTime, "RunningTime=", time() - self.StartTime, "/", self.RunningTime
            self.model.stop_training = True

        # Looks like on_epoch_sart is not called!
        self.EpochStartTime = time()


class GracefulExit(keras.callbacks.Callback):
    def __init__(self):
        super(GracefulExit, self).__init__()
        self.Stop = False
        self.sig1 = signal.getsignal(signal.SIGINT)
        self.sig2 = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.calls = 0

    def on_train_begin(self, batch, logs={}):
        signal.signal(signal.SIGINT, self.signal_handler)
        # signal.signal(signal.SIGTERM, self.signal_handler)

    def on_train_end(self, batch, logs={}):
        signal.signal(signal.SIGINT, self.sig1)
        # signal.signal(signal.SIGTERM, self.sig2)

    def kill_child_processes(self, signum=signal.SIGKILL):
        parent_id = os.getpid()
        ps_command = sp.Popen("pgrep -P %d" % parent_id, shell=True, stdout=subprocess.PIPE)
        ps_output = ps_command.stdout.read()
        retcode = ps_command.wait()
        for pid_str in ps_output.strip().split("\n")[:-1]:
            os.kill(int(pid_str), signum)

    # Drastic Actions... otherwise the other processes would get stuck.
    def KillMinusNine(self):
        PID = os.getpgid(0)
        PID1 = os.getpid()
        print "Killing self -9. PID:", PID, PID1
        try:
            if PID1 != PID:
                self.kill_child_processes()
                os.kill(PID1, signal.SIGKILL)
            else:
                os.kill(PID1, signal.SIGKILL)
        except:
            print "Couldn't kill:", PID, PID1
            pass

    def signal_handler(self, signal, frame):
        self.calls += 1
        if sys.flags.interactive:
            if self.calls == 2:
                self.Stop = True
                print
                print('GracefulExit Callback: You pressed Ctrl+C or sent kill signal twice!')
                print('GracefulExit Callback: Will stop at end of Epoch.')
                print(
                    'GracefulExit Callback: Press Ctrl+C or send kill signal again in this epoch to stop immediately.')
            elif self.calls > 2:
                print
                print('GracefulExit Callback: You pressed Ctrl+C or sent kill signal three times!')
                print "GracefulExit Callback: Attempting exit."
                self.KillMinusNine()
            else:
                print
                print('You pressed Ctrl+C or sent kill signal!')
                print('Press Ctrl+C or send kill signal again in this epoch to stop at end of this epoch.')
                print('Press Ctrl+C or send kill signal again in this epoch to stop immediately.')
        else:
            if self.calls == 2:
                print
                print('GracefulExit Callback: You pressed Ctrl+C or sent kill signal twice.')
                print "GracefulExit Callback: Attempting exit."
                self.KillMinusNine()
            else:
                print
                print('GracefulExit Callback: You pressed Ctrl+C or sent kill signal. Stopping at end of epoch.')
                self.Stop = True

    def on_epoch_end(self, batch, logs={}):

        if self.Stop:
            print
            print "GracefulExit Callback: Stopping training."
            self.model.stop_training = True
        self.calls = 0


class ROCModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, filepath, verbose=True):
        super(Callback, self).__init__()
        self.verbose = verbose
        self.filepath = filepath
        self.best = 0.0

    def on_epoch_end(self, epoch, logs={}):
        self.X, self.y, self.weights, _ = self.model.validation_data
        fpr, tpr, _ = roc_curve(self.y, self.model.predict(self.X, verbose=self.verbose).ravel(),
                                sample_weight=self.weights)
        select = (tpr > 0.1) & (tpr < 0.9)
        current = auc(tpr[select], 1 / fpr[select])

        if current > self.best:
            if self.verbose > 0:
                print("Epoch %05d: %s improved from %0.5f to %0.5f, saving model to %s"
                      % (epoch, 'AUC', self.best, current, self.filepath))
            self.best = current
            self.model.save_weights(self.filepath, overwrite=True)
        else:
            if self.verbose > 0:
                print("Epoch %05d: %s did not improve (current = %0.5f)" % (epoch, 'AUC', current))
