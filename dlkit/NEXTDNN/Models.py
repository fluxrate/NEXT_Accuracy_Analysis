from DLTools.ModelWrapper import *

from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import  BatchNormalization,Dropout,Flatten, Input
from keras.layers.pooling import *
from keras.layers.convolutional import Conv3D
from keras.models import model_from_json

class Fully3DImageClassification(ModelWrapper):
    def __init__(self, Name, input_shape, width=(5, 10, 30), depth=0, BatchSize=2048,
                 N_classes=100, init=0, BatchNormalization=False, Dropout=False,
                 NoClassificationLayer=False,
                 activation='relu',**kwargs):

        super(Fully3DImageClassification, self).__init__(Name,**kwargs)

        self.width=width
        self.depth=depth
        self.input_shape=input_shape
        self.N_classes=N_classes
        self.init=init

        self.Dropout=Dropout

        self.BatchSize=BatchSize
        self.BatchNormalization=BatchNormalization
        self.Activation=activation
        self.NoClassificationLayer=NoClassificationLayer
        
        self.MetaData.update({ "width":self.width,
                               "depth":self.depth,
                               "Dropout":self.Dropout,
                               "BatchNormalization":BatchNormalization,
                               "input_shape":self.input_shape,
                               "N_classes":self.N_classes,
                               "init":self.init})
    def Build_FARBIN(self):
        input=Input(self.input_shape[1:])
        modelT = Flatten(input_shape=self.input_shape)(input)

#        model.add(Dense(self.width,init=self.init))
        modelT = (Activation('relu')(modelT))

        for i in xrange(0,self.depth):
            if self.BatchNormalization:
                modelT=BatchNormalization()(modelT)

            modelT=Dense(self.width,kernel_initializer=self.init)(modelT)
            modelT=Activation(self.Activation)(modelT)

            if self.Dropout:
                modelT=Dropout(self.Dropout)(modelT)

        if not self.NoClassificationLayer:
            modelT=Dense(self.N_classes, activation='softmax',kernel_initializer=self.init)(modelT)

        self.inputT=input
        self.modelT=modelT
        
        self.Model=Model(input,modelT)

    #Creates a model based on sequential from Keras, and adds layers based on widths defined
    def Build(self):
        input = Input(self.input_shape[1:]) #first layer input
        #add dense layers to the model, with activation in between each

        # need to fix dense layer dimension to not be hardcoded...
        
        # myCustomModel = (Dense(self.width[0]))(input) #second layer Dense
        myCustomModel = (Dense(5))(input) #second layer Dense
        myCustomModel = (Activation('relu'))(myCustomModel)
        
        #myCustomModel = (Dense(self.width[1]))(myCustomModel)
        myCustomModel = (Dense(10))(myCustomModel)
        myCustomModel = (Activation('softmax'))(myCustomModel)
        
        # myCustomModel = (Dense(self.width[2]))(myCustomModel)
        myCustomModel = (Dense(30))(myCustomModel)
        myCustomModel = (Activation('relu'))(myCustomModel)

        #Downscales from the input shape to a size of (100,100,100) with a pooling layer...
        #this may be IN ERROR due to bin'ing of data prior to model via histogram methods

        # myCustomModel = (MaxPooling3D(pool_size=(2,2,2), strides = None))(myCustomModel)

        #Add a 3D convolutional net
        myCustomModel = (Conv3D(15, (3,3,3), data_format='channels_first', padding = 'same'))(myCustomModel)

        #Add a dropout layer to help prevent overtraining
        myCustomModel = (Dropout(0.1))(myCustomModel)

        #Add another 3D convolutional net
        myCustomModel = (Conv3D(30, (3,3,3), activation=self.activation))(myCustomModel)

        #Add an activation in between
        myCustomModel = (Activation('relu'))(myCustomModel)
        
        #Add another dropout layer
        myCustomModel = (Dropout(0.05))(myCustomModel)
        
        if not self.NoClassificationLayer:
            myCustomModel = (Dense(self.N_classes, activation='softmax',kernel_initializer=self.init))(myCustomModel)
        # Don't have any idea what these do...
        # Need to clarify . . .
        #
        self.inputT=input
        self.modelT=myCustomModel

        #sets the model defined in build as the model
        #self.Model = Model(input, myCustomModel)

        self.Model = Model(inputs=input, outputs=myCustomModel)                    





#
# Leave this alone
#

class MergerModel(ModelWrapper):
    def __init__(self, Name, Models, N_Classes, init, **kwargs):
        super(MergerModel, self).__init__(Name,**kwargs)
        self.Models=Models
        self.N_Classes=N_Classes
        self.init=init
        
    def Build(self):

        MModels=[]
        MInputs=[]
        for m in self.Models:
            MModels.append(m.modelT)
            MInputs.append(m.inputT)
        if len(self.Models)>0:
            print "Merged Models"
            modelT=concatenate(MModels)#(modelT)
            
        modelT=Dense(self.N_Classes, activation='softmax',kernel_initializer=self.init)(modelT)
        

        self.modelT=modelT
        
        self.Model=Model(MInputs,modelT)

                
