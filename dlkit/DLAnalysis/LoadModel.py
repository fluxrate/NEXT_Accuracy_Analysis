from DLTools.ModelWrapper import *


def LoadModels(ModelDirs, verbose=False, returndict=False, MetaDataOnly=False):
    if returndict:
        MyModels = {}
    else:
        MyModels = []

    for i, ModelDir in enumerate(ModelDirs):
        try:
            if verbose:
                print "Loading Model number", i, "from:", ModelDir
            if ModelDir[-1] == "/": ModelDir = ModelDir[:-1]
            MyModel = ModelWrapper(Name=os.path.basename(ModelDir),
                                   InDir=os.path.dirname(ModelDir))
            MyModel.Load(ModelDir, MetaDataOnly=MetaDataOnly)
            if returndict:
                MyModels[os.path.basename(ModelDir)] = MyModel
            else:
                MyModels.append(MyModel)
        except:
            print "Failed to load model from:", ModelDir

    return MyModels


if __name__ == '__main__':
    import sys, glob

    if len(sys.argv) < 2:
        print "Error: No Model directory/directories specified."

    LoadModel = sys.argv[1]
    ModelDirs = []
    if len(sys.argv) == 2:
        if LoadModel[-1] == "/":
            ModelDirs = glob.glob(LoadModel + "*")
        else:
            ModelDirs = [LoadModel]
    else:
        ModelDirs = sys.argv[1:]

    MyModels = LoadModels(ModelDirs, True)
    print "Loaded Models into MyModels list."
