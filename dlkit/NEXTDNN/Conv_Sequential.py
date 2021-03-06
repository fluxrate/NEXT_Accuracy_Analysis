from DLTools.ModelWrapper import ModelWrapper
from keras.layers import Input, Dense, Layer,BatchNormalization,Dropout,Flatten,pooling,Activation
import keras.models 
from keras.layers.convolutional import Conv2D
from keras.models import Sequential


class Conv_Sequential(ModelWrapper):
    def __init__(self,Name,
                 InputShape,
                 Depth = 0,
                 Widths = 0,
                 BatchSize= 0,
                 Activation = 'relu',
                 layers = None,
                 Loss = "mean_squared_logarithmic_error",
                 Optimizer = 'SGD',
                 Metrics = 'accuracy'
                  ):
        super(Conv_Sequential,self).__init__(Name,Loss,Optimizer)
        
        self.InputShape = self.MetaData["InputShape"] = InputShape
        self.Depth = self.MetaData["Depth"]= Depth
        self.Widths = self.MetaData["Widths"] = Widths
        self.Activation = self.MetaData["Activation"] = Activation
        self.BatchSize = self.MetaData["BatchSize"] = BatchSize
 
 #Creates a model based on sequential from Keras, and adds layers based on widths defined 
    def Build(self):
        model = Sequential()
        model.add(Activation('relu',input_shape = self.InputShape))
        model.add(Conv2D(self.BatchSize,(1,1),data_format='channels_first',input_shape=self.InputShape,padding= 'same'))
        model.add(Dropout(0.1))
        model.add(Flatten()) #had to add a flatten or the NEXT_Experiment training would give a value error
        #Cycles through the list of widths for the model and adds layers
        for i in xrange(0,self.Depth):
            model.add(Dense(self.Widths,input_shape = self.InputShape))
        #model.add(Conv2D(32,(10,10)))
        #model.add(Activation('relu'))        
        #Downscales from the input shape to a size of (100,100,100)
        #model.add(pooling.MaxPooling3D(pool_size=(2, 2, 2),strides = None))
        #drop out of 10% after the first convolutional net
        #model.add((Conv3D(16,(3,3,3),activation=self.Activation)))
        #model.add(Dropout(.05))
        #drop out of 5% after the 2nd convolutional net
        self.Model = model
        #sets the model defined in build as the model
        
        
    
   