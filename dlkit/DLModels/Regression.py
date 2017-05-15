from DLTools.ModelWrapper import *

from keras.models import Sequential
from keras.layers.core import Dense, Activation, MaxoutDense, Dropout, Highway
from keras.layers import BatchNormalization, Dropout
from keras.models import model_from_json
from keras.regularizers import l1l2, l1
from DLModels.Loss import GaussianNLL, GaussianMSE
from keras.optimizers import Adam


class FullyConnectedRegression(ModelWrapper):
    def __init__(self, Name, N_input=0, width=0, depth=0, init=0, outputsize=1):

        super(FullyConnectedRegression, self).__init__(Name)

        self.width = width
        self.depth = depth
        self.N_input = N_input
        self.init = init
        self.outputsize = outputsize

        self.MetaData.update({"width": self.width,
                              "depth": self.depth,
                              "N_input": self.N_input,
                              "init": self.init})

    def Build(self):
        model = Sequential()
        model.add(Dense(self.width, input_dim=self.N_input, init=self.init))

        model.add(Activation('tanh'))

        for i in xrange(0, self.depth):
            model.add(BatchNormalization())
            model.add(Dense(self.width, init=self.init))
            model.add(Activation('tanh'))
            model.add(Dropout(0.5))

        model.add(Dense(self.outputsize, input_dim=self.width))

        self.Model = model

    def Compile(self, Loss="GaussianNLL", Optimizer=None):

        if Optimizer == "Adam":
            lrinit = 1.e-5
            Optimizer = Adam(lr=lrinit, beta_1=0.99, beta_2=0.999, epsilon=1e-08)
        if Loss == "GaussianNLL":
            self.Model.compile(optimizer=Optimizer, loss=GaussianNLL, metrics=[GaussianMSE])
        else:
            self.Model.compile(loss=Loss, optimizer=Optimizer, metrics=["accuracy"])


class FullyConnectedClassification(ModelWrapper):
    def __init__(self, Name, N_input=0, width=0, depth=0, N_classes=100, init=0):
        super(FullyConnectedClassification, self).__init__(Name)

        self.width = width
        self.depth = depth
        self.N_input = N_input
        self.N_classes = N_classes
        self.init = init

        self.MetaData.update({"width": self.width,
                              "depth": self.depth,
                              "N_input": self.N_input,
                              "N_classes": self.N_classes,
                              "init": self.init})

    def Build(self):
        model = Sequential()
        model.add(Dense(self.width, input_dim=self.N_input, init=self.init))

        model.add(Activation('relu'))

        for i in xrange(0, self.depth):
            model.add(BatchNormalization())
            model.add(Dense(self.width, init=self.init))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))

        model.add(Dense(self.N_classes, activation='softmax'))

        self.Model = model

    def Compile(self, Loss="categorical_crossentropy", Optimizer="rmsprop"):
        self.Model.compile(loss=Loss, optimizer=Optimizer, metrics=["accuracy"])


# Clipped from: https://github.com/mickypaganini/dark-mem/blob/master/toolkit/layers.py

from keras import activations, initializations, regularizers, constraints
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.callbacks import Callback
from keras.layers import Dense

from sklearn.metrics import roc_curve, auc

import numpy as np


def custom_uniform(shape, range=(-1, 1), name=None):
    min_, max_ = range
    return K.variable(np.random.uniform(low=min_, high=max_, size=shape), name=name)


class ParametricDense(Layer):
    def __init__(self, output_dim, basis_dim, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, basis_constraint=None,
                 bias=True, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.basis_dim = basis_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.basis_constraint = constraints.get(basis_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(ParametricDense, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.init((self.output_dim, input_dim, self.basis_dim), name='{}_W'.format(self.name))

        self.basis = self.init((self.output_dim, self.basis_dim), name='{}_basis'.format(self.name))

        if self.bias:
            self.b = K.zeros((self.output_dim, self.basis_dim), name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.basis, self.b]
        else:
            self.trainable_weights = [self.W, self.basis]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint
        if self.basis_constraint:
            self.constraints[self.basis] = self.basis_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):

        Z = K.dot(x, self.W)

        if self.bias:
            Z += self.b

        def step(x, states):
            return K.softmax(x), []

        output = K.sum(K.rnn(step, Z, [])[1] * self.basis, axis=-1)

        return output

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'basis_dim': self.basis_dim,
                  'basis_constraint': self.basis_constraint.get_config() if self.basis_constraint else None}

        base_config = super(ParametricDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ROCModelCheckpoint(Callback):
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


from keras.constraints import Constraint


class Bounded(Constraint):
    '''
    Constrain the weights to be in [lower, upper]
    '''

    def __init__(self, lower=0., upper=None, hard_lower=False, hard_upper=False):
        self.lower = lower
        self.upper = upper if upper else np.inf
        self.hard_upper = hard_upper
        self.hard_lower = hard_lower

    def __call__(self, p):

        if self.hard_lower:
            p -= (K.min(p) + self.lower)
        if self.hard_upper:
            p /= K.max(p)
        p = K.clip(p, self.lower, self.upper)
        return p


class HighwayRegression(ModelWrapper):
    def __init__(self, Name, N_input=0, width=0, depth=0, init=0, n_highways=10):
        super(HighwayRegression, self).__init__(Name)

        self.width = width
        self.depth = depth
        self.N_input = N_input
        self.init = init
        self.n_highways = n_highways
        self.CustomObjects = {'ParametricDense': ParametricDense}

        self.MetaData.update({"width": self.width,
                              "depth": self.depth,
                              "N_input": self.N_input,
                              "init": self.init,
                              "n_highways": self.n_highways})

    def Build(self):
        # Clipped from: https://github.com/mickypaganini/dark-mem/blob/master/train_x%2BDLmem_hardtarget.py

        net = Sequential()

        # Leave these parameters hard-coded for now.
        net.add(MaxoutDense(96, 10, input_dim=self.N_input,
                            init=self.init,
                            W_regularizer=l1(0.001)))
        # net.add(Dropout(0.2))


        for _ in xrange(self.n_highways):
            net.add(Highway(activation='relu', transform_bias=-5,
                            init=self.init,
                            W_regularizer=l1l2(0.0001, 0.0001)))

            # net.add(Dropout(0.2))

        net.add(Dense(64, activation='relu'))
        # net.add(Dropout(0.1))
        net.add(Dense(40, activation='relu'))

        HARD_UPPER = True
        HARD_LOWER = False

        def _initializer(shape, name):
            return custom_uniform(shape, (0, 1), name)

        net.add(ParametricDense(1, basis_dim=15,
                                init=_initializer,
                                basis_constraint=Bounded(0., 1.5, hard_lower=HARD_LOWER, hard_upper=HARD_UPPER)))

        self.Model = net
