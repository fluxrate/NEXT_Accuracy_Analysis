# ModelWrapper.py
# Amir Farbin

# import sys
import os
import cPickle as pickle
from keras.models import model_from_json
from keras.optimizers import *  # optimizer_from_config
if "optimizer_from_config" not in dir():
    from keras.optimizers import deserialize as optimizer_from_config

class ModelWrapper(object):
    """
    ModelWrapper is used to faciliate the loading of Keras models with additional meta data.
    """

    def __init__(self, Name, Loss=None, Optimizer=None, InDir=False, LoadPrevious=False, OutputBase="TrainedModels"):
        self.Name = Name
        self.Loss = Loss
        self.Optimizer = Optimizer
        self.OptimizerClass=None
        self.MetaData = {"Name": Name, "Optimizer": Optimizer, "Loss": Loss, "InputMetaData": []}
        self.Model = None
        self.LoadPrevious = LoadPrevious
        # Should this be a None instead of False?
        if InDir: self.InDir = InDir
        self.LoadPrevious = LoadPrevious
        self.OutputBase = OutputBase
        self.Metrics=[]
        self.Initialize()

    def Initialize(self, Overwrite=False):
        try:
            os.mkdir(self.OutputBase)
        except:
            pass
        # TODO shouldn't this be done in __init__?
        self.OutDir = os.path.join(self.OutputBase, self.Name)
        self.InDir = self.OutDir

        PreDirs = []

        if not Overwrite:
            # TODO should be we doing this with a timestamp instead?
            i = 1
            OutDir = self.OutDir

            self.PreviousOutDir = OutDir

            while os.path.exists(OutDir):
                PreDirs.append(OutDir)
                OutDir = self.OutDir + "." + str(i)
                i += 1

            self.OutDir = OutDir

            if self.LoadPrevious and not self.OutDir == self.PreviousOutDir:
                Loaded = False

                while not Loaded and len(PreDirs) != 0:
                    try:
                        self.PreviousOutDir = PreDirs.pop()
                        print "Loading Previous Model From:", self.PreviousOutDir
                        self.Load(self.PreviousOutDir, Initialize=False)
                        Loaded = True
                    except:
                        print "Failed to load from: ", self.InDir

        self.MetaData["OutDir"] = self.OutDir

    def Save(self, OutDir=False):
        if OutDir:
            self.OutDir = OutDir

        try:
            os.makedirs(self.OutDir)
        except:
            # TODO log error
            pass
            # print "Error making output Directory"

        try:
            self.MetaData["History"] = self.History.history
        except:
            pass

        with open(os.path.join(self.OutDir, "Model.json"), "w") as tmp_file:
            tmp_file.write(self.Model.to_json())
        self.Model.save_weights(os.path.join(self.OutDir, "Weights.h5"), overwrite=True)

        try:
            self.MetaData["OptimizerConfig"] = self.Optimizer.get_config()
            self.MetaData["OptimizerClass"] = self.OptimizerClass
        except:
            # TODO log warning?
            pass

        self.MetaData["Optimizer"] = self.OptimizerClass
        self.MetaData["Loss"] = self.Loss
        pickle.dump(self.MetaData, open(os.path.join(self.OutDir, "MetaData.pickle"), "wb"))

    def MakeOutputDir(self):
        # TODO should this be a static function?
        try:
            os.mkdir(self.OutDir)
        except:
            pass

    def Load(self, InDir=False, MetaDataOnly=False, Overwrite=False, Initialize=True):
        if InDir:
            self.InDir = InDir

        if not MetaDataOnly:
            try:
                self.Model = model_from_json(open(os.path.join(self.InDir, "Model.json"), "r").read(),
                                             custom_objects=self.CustomObjects)
            except:
                self.Model = model_from_json(open(os.path.join(self.InDir, "Model.json"), "r").read())

            self.Model.load_weights(os.path.join(self.InDir, "Weights.h5"))

        # TODO does this automatically close the file too? If not, should use a 'with' statement
        MetaData = pickle.load(open(os.path.join(self.InDir, "MetaData.pickle"), "rb"))
        self.MetaData.update(MetaData)

        OldMD = dict(MetaData)
        OldMD["InputMetaData"] = []
        try:
            self.MetaData["InputMetaData"].append(OldMD)
        except:
            self.MetaData["InputMetaData"] = [OldMD]

        if "Metrics" in self.MetaData:
            self.Metrics = self.MetaData["Metrics"]

        self.MetaData["InputDir"] = self.InDir

        if MetaDataOnly:
            return

        if "OptimizerClass" in self.MetaData.keys():
            self.OptimizerClass = self.MetaData["OptimizerClass"]
            # Configure the Optimizer, using optimizer configuration parameter.
            try:
                self.Optimizer = optimizer_from_config({"class_name": self.MetaData["OptimizerClass"],
                                                        "config": self.MetaData["OptimizerConfig"]})
            except:
                print "Warning: Failed to instantiate optimizer. Trying again."
                opt_Instance = eval(self.MetaData["OptimizerClass"])
                opt_config = opt_Instance.get_config()
                for param in opt_config:
                    try:
                        opt_config[param] = self.MetaData[param]
                    except Exception as detail:
                        print "Warning: optimizer configuration parameter", param,
                        print "was not set in saved model. Will use default."
                self.Optimizer = optimizer_from_config({"class_name": self.MetaData[:OptimizerClass:],
                                                        "config": opt_config})

        else:
            self.Optimizer = self.MetaData["Optimizer"]

        if "Loss" in self.MetaData.keys():
            self.Loss = self.MetaData["Loss"]

        if Initialize:
            self.Initialize(Overwrite=Overwrite)

    def Compile(self, Loss=False, Optimizer=False, Metrics=[]):
        if Loss:
            self.Loss = Loss
        if Optimizer:
            self.Optimizer = Optimizer
        if len(Metrics) > 0:
            self.Metrics = Metrics
            self.MetaData["Metrics"] = Metrics

        self.Model.compile(loss=self.Loss, optimizer=self.Optimizer, metrics=self.Metrics)

    def Train(self, X, y, Epochs, BatchSize, Callbacks=[], validation_split=0.):
        History = self.Model.fit(X, y, nb_epoch=Epochs, batch_size=BatchSize,
                                 callbacks=Callbacks, validation_split=validation_split)
        self.History = History
        self.MetaData["History"] = History.history

    def BuildOptimizer(self, optimizer, config):
        # Configure the Optimizer, using optimizer configuration parameter.
        self.OptimizerClass = optimizer
        try:
            opt_Instance = eval(optimizer + "()")
            opt_config = opt_Instance.get_config()
            for param in opt_config:
                try:
                    opt_config[param] = config[param]
                except Exception as detail:
                    print "Warning: optimizer configuration parameter", param,
                    print "was not set in configuration file. Will use default."
            optimizer = optimizer_from_config({"class_name": optimizer, "config": opt_config})
            self.Optimizer = optimizer
        except Exception as detail:
            print "Error:", detail
            print "Warning: unable to instantiate and configure optimizer", optimizer,
            print ". Will attempt to use default config."
            self.Optimizer = optimizer

    def Build(self):
        pass
