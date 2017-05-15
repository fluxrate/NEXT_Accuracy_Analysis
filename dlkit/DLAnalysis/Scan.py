import tabulate

#______________________________________________________________________________
def ScanTable(Models, Params, sortI=[0]):
    aTable = []

    columns = Params
    for m in Models:

        if type(m) == str:
            Model = Models[m]
        else:
            Model = m

        row = [Model.Name]
        for col in columns[1:]:
            if "[" in col:
                colsplit = col.split("[")
                coln = int(colsplit[-1][:-1])
                try:
                    data = Model.MetaData[colsplit[0]][coln]
                except:
                    data = "N/A"
            else:
                try:
                    data = Model.MetaData[col]
                except:
                    data = "N/A"

            row += [data]

        aTable += [row]

    if sortI:
        print tabulate.tabulate(sorted(aTable, key=lambda a: tuple(map(lambda I: a[I], sortI))), headers=columns,
                                floatfmt=".4f")
    else:
        print tabulate.tabulate(aTable)


#______________________________________________________________________________
def PlotMetaData(Models, keys, mode="epoch", loc="center left", label="", **kwargs):
    import matplotlib.pyplot as plt

    assert keys
    if not isinstance(keys, list):
        keys = [keys]

    def PullValue(m):
        d = m.MetaData
        for k in keys:
            d = d[k]
        return d

    R = map(PullValue, Models)

    plotted = False
    for i, r in enumerate(R):
        if isinstance(r, list):
            if mode == "epoch":
                try:
                    plt.plot(range(len(r)), r, label=Models[i].Name+label, **kwargs)
                    plotted = True
                except:
                    pass
            elif mode == "hist":
                plt.hist(r, label=Models[i].Name+label, **kwargs)
                plotted = True
            if plotted:
                ax = plt.gca()
                box = ax.get_position()
                # ax.set_position([box.x0, box.y0, box.y1 * 0.8, box.x1])
                plt.legend(loc=loc, bbox_to_anchor=(1, .5), fontsize=8)

    if not plotted:
        x = range(len(R))
        plt.bar(x, R, **kwargs)

        axes = plt.gca()
        miny = min(R)
        maxy = max(R)

        if miny < 0:
            miny = miny * 1.1
        else:
            miny = miny * 0.9

        if maxy < 0:
            maxy = maxy * 0.9
        else:
            maxy = maxy * 1.1

        axes.set_ylim([miny, maxy])

        plt.xticks(x, map(lambda m: m.Name, Models), rotation='vertical')

    # Change Names of Models


#______________________________________________________________________________
def ResetNames(Models, Params, Func=None):
    for m in Models:
        Name = ""
        if Func:
            Name=Func(Params)
        else:
            for p in Params:
                try:
                    Name += p + "=" + str(m.MetaData[p]) + " "
                except:
                    Name = m.Name
        m.Name = Name


#______________________________________________________________________________
def GetEpochs(MyModels):
    """
    Add the number of epochs to MetaData by counting length of history
    """
    def GetEpoch(m):
        try:
            k = m.MetaData["History"].keys()[0]
            m.MetaData["Epochs"] = len(m.MetaData["History"][k])
            for t in m.MetaData["InputMetaData"]:
                m.MetaData["Epochs"] += len(t["History"][k])
        except:
            pass
    tmp = map(GetEpoch, MyModels)


#______________________________________________________________________________
def GetHistorical(Models, Params=[]):
    """
    Pull data from previous trainings into latest MetaData, with "All_" suffix
    """
    if Params == []:
        N = Models[0].MetaData["History"].keys()
        for n in N:
            Params.append(["History", n])

        Params += GetGoodParams(Models)

    def PullValue(D, keys):
        d = dict(D)
        for k in keys:
            try:
                d = d[k]
            except:
                pass
        return d

    NewParams = set()
    for m in Models:
        for p in Params:
            Vals = []
            try:
                for i, t in enumerate(m.MetaData["InputMetaData"][1:]):
                    if type(p) == list:
                        Val = PullValue(t, p)
                    else:
                        Val = t[p]

                    if type(Val) == list:
                        Vals += Val
                    else:
                        Vals.append(Val)

                    if type(p) == list:
                        pName = "All_" + reduce(lambda x, y: x + "." + y, p)
                        m.MetaData[pName] = Vals
                    else:
                        pName = "All_" + p
                        m.MetaData[pName] = Vals

                    NewParams.add(pName)
            except:
                pass
    return list(NewParams)


#______________________________________________________________________________
def GetHistoricalOld(Models, Params):
    for m in Models:
        for p in Params:
            # Vals=[m.MetaData[p]]
            Vals = []
            for t in m.MetaData["InputMetaData"]:
                Vals.append(t[p])
            m.MetaData["All_" + p] = Vals


#______________________________________________________________________________
def GetGoodParams(MyModels):
    import numpy as np
    GoodTypes = [int, float, np.float32, np.float64]
    Params = []
    for k in MyModels[0].MetaData:
        MType = type(MyModels[0].MetaData[k])
        if MType in GoodTypes:
            Params.append(k)
    return Params


#______________________________________________________________________________
def PlotMetaDataMany(inModels, N, keys, select=None, sort=False, switch=False, **kwargs):
    import matplotlib.pyplot as plt
    if select:
        sModels=map(select,inModels)
    else:
        sModels=list(inModels)

    if sort:
        Models=sorted(sModels,key=lambda m: tuple(map(lambda p: m.MetaData[p],sort)))
    else:
        Models=list(sModels)
        
    Ms=[]
    while len(Models)!=0:
        if not switch:
            while len(Models)>0 and len(Ms)<N:
                Ms.append(Models.pop(0))
        else:
            first=True
            vals=[]

            i=0
            while len(Models)>0 and len(Ms)<N and i<len(Models):
                #print i, len(Models),len(Ms),vals
                if first:
                    for s in switch:
                        vals.append(Models[i].MetaData[s])
                    first=False
                    Ms.append(Models.pop(i))
                else:
                    for ii,s in enumerate(switch):
                        Test=True
                        if vals[ii] != Models[i].MetaData[s]:
                            #print s,vals[ii],Models[i].MetaData[s],"  ",
                            Test=False
                            break
                        #print
                    if Test:    
                        Ms.append(Models.pop(i))
                    else:
                        i+=1

        if type(keys)==list:
            for k in keys:
                PlotMetaData(Ms, k, label=" "+reduce(lambda x,y: x+" "+y,k),**kwargs)
        else:
            PlotMetaData(Ms, keys, **kwargs)
            
        plt.show()
        Ms=[]


#______________________________________________________________________________
def SelectModels(Models,func):
    out=[]
    for m in Models:
        try:
            if func(m):
                out.append(m)
        except:
            pass

    return out


#______________________________________________________________________________
def EvalMetaData(model,expression,params=[]):
    P=GetGoodParams([model])
    P+=params
    for p in P:
        exec(p+"="+str(model.MetaData[p]))

    return eval(expression)


#______________________________________________________________________________
def ModelEvaluator(model,params=[]):
    def f(expression):
        return EvalMetaData(model,expression,params)
    return f


#______________________________________________________________________________
def MetaDataEvaluator(expression,params=[]):
    def f(model):
        return EvalMetaData(model,expression,params)
    return f


#______________________________________________________________________________
if __name__ == '__main__':
    from DLAnalysis.LoadModel import *
    import argparse
    import numpy as np
    import sys, glob

    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--sortbycolumn', default="0",
                        help="Camma separated list of indecies in quotes. Sort rows based on these column numbers.")
    parser.add_argument("files", nargs="*")

    parser.add_argument('-c', '--columnorder', default="0",
                        help="Camma separated list of indecies in quotes. Specifies order that columns are displayed.")

    args = parser.parse_args()

    sI = args.sortbycolumn.split(",")
    sortI = map(lambda x: int(x), sI)

    LoadModel = args.files[0]
    if len(args.files) == 1:
        if LoadModel[-1] == "/":
            ModelDirs = glob.glob(LoadModel + "*")
        else:
            ModelDirs = [LoadModel]
    else:
        ModelDirs = args.files

    MyModels = LoadModels(ModelDirs, False, True, True)


    def GetEpochs(m):
        try:
            MyModels[m].MetaData["Epochs"] = len(MyModels[m].MetaData["History"]["val_loss"])
        except:
            print "Failed to get history.",m


    map(GetEpochs, MyModels)

    FirstModel = MyModels[MyModels.keys()[0]]

    GoodTypes = [int, float, np.float32, np.float64]
    Params = ["Model Name"]
    for k in FirstModel.MetaData:
        MType = type(FirstModel.MetaData[k])
        if MType in GoodTypes:
                Params.append(k)

    Params.append("FinalScore[0]")
    Params.append("FinalScore[1]")

    Params.append("InitialScore[0]")
    Params.append("InitialScore[1]")

    Params1 = list(Params)
    sortI1 = list(sortI)
    if args.columnorder != "0":
        cI = args.columnorder.split(",")
        columnI = map(lambda x: int(x), cI)
        for I1, I2 in enumerate(columnI):
            Params1[I1] = Params[I2]

            # for I1,I2 in enumerate(sortI):
            #    print I1,I2
            #    sortI1[I1]=sortI[sortI[I1]]

    ScanTable(MyModels, Params1, sortI1)

# python -im DLAnalysis.Scan -c "0,2,3,6,1,4,5,7" -s "1,2,0" TrainedModels/
# dict(map( lambda m: (m,len(MyModels[m].MetaData["History"]["val_loss"])), MyModels))
