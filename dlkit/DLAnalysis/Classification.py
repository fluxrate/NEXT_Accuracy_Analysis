# Analysis
import numpy as np

import matplotlib as mpl

mpl.use('pdf')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

mpColors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "white"]+["blue", "green", "red", "cyan", "magenta", "yellow", "black", "white"]
#mpColors = map( lambda x: "C"+str(x), range(100))
            

def BinData(Xs, Y, Ymin, Ymax, Ybins):
    out = []
    Ystep = (Ymax - Ymin) / Ybins
    Ysample = np.arange(Ymin, Ymax, Ystep)
    for X in Xs + [Y]:
        Xout = []
        for y in Ysample:
            Xout.append(X[np.where((Y >= y) & (Y < y + Ystep))])
        out.append(Xout)
    return out + [Ysample]


def BinDataIndex(Y, Ymin, Ymax, Ybins):
    out = []
    Ystep = (Ymax - Ymin) / Ybins
    Ysample = np.arange(Ymin, Ymax, Ystep)
    for y in Ysample:
        out.append(np.where((Y >= y) & (Y < y + Ystep)))

    return out, Ysample


def MultiClassificationAnalysis(MyModel, Test_X=[], Test_Y=[], BatchSize=1024, PDFFileName=False,
                                IndexMap=False, MakePlot=True, result=False):
    if type(result) == bool:
        result = MyModel.Model.predict(Test_X, batch_size=BatchSize)

    # Unfortunately, we can use the generator, because we need Test_Y
    # result = MyModel.Model.predict_generator(validation_generator, batch_size=BatchSize)

    NClasses = Test_Y.shape[1]
    MetaData = {}

    MyIndexMap=IndexMap
    if IndexMap:
        if type(IndexMap.keys()[0])==str:
            MyIndexMap={}
            for k in IndexMap:
                MyIndexMap[IndexMap[k]]=k

    for ClassIndex in xrange(0, NClasses):
        fpr, tpr, _ = roc_curve(Test_Y[:, ClassIndex],
                                result[:, ClassIndex])
        roc_auc = auc(fpr, tpr)

        lw = 2

        if MyIndexMap:
            ClassName = MyIndexMap[ClassIndex]
        else:
            ClassName = "Class " + str(ClassIndex)

        MetaData[ClassName + "_AUC"] = roc_auc

        if MakePlot:
            plt.plot(fpr, tpr, color=mpColors[ClassIndex],
                     lw=lw, label=ClassName + ' (area = %0.2f)' % roc_auc)

            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])

            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')

            plt.legend(loc="lower right")

    if MakePlot and PDFFileName:
        #try:
        plt.savefig(MyModel.OutDir + "/" + PDFFileName + ".pdf")
        #except:
        #    print "Warning: Unable to write out:", MyModel.OutDir + "/" + PDFFileName + ".pdf"

    return result, MetaData


def BinMultiClassificationAnalysis(Model, Test_Y, Y_binning, bin_indecies, result, IndexMap=False):
    BinnedResults = {}

    for i, E in enumerate(Y_binning):
        tmp, NewMetaData = MultiClassificationAnalysis(Model, Test_Y=Test_Y[bin_indecies[i]],
                                                       result=result[bin_indecies[i]],
                                                       MakePlot=False, IndexMap=IndexMap)

        for MD in NewMetaData:
            if MD in BinnedResults:
                BinnedResults[MD].append(NewMetaData[MD])
            else:
                BinnedResults[MD] = [NewMetaData[MD]]

    for MD in BinnedResults:
        plt.plot(Y_binning, BinnedResults[MD], label=MD)
    plt.legend(loc="center left", bbox_to_anchor=(1, .5), fontsize=8)

    return BinnedResults
