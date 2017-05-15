#
# Gaussian Loss functions from Peter Sadowski.
# Gaussian Loss Modified from Original https://groups.google.com/forum/#!searchin/keras-users/output$20distribution/keras-users/caIEicDknlU/ut7nch8bDAAJ
# 

from keras import backend as K
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.layers.recurrent import LSTM
from keras.models import Sequential

import theano
from theano import tensor as T

import numpy as np

stability_factor = 1e-6


def GaussianNLL(y, parameters):
    '''
    Compute NLL for Gaussian model with mean and std..
    y = Nx1 vector of target (data) values.
    parameters = N x 2 matrix. 
    '''
    # M = 1
    mean = parameters[:, 0]
    sigma = parameters[:, 1]
    sigma = K.exp(sigma)

    ymean = y[:, 0]  # Other columns are ignored.

    # Log space.
    term1 = -K.log(1.0 / (np.sqrt(2 * np.pi) * sigma + stability_factor))
    term2 = (mean - ymean) ** 2 / (sigma ** 2 * 2 + stability_factor)
    nll = term1 + term2
    rval = K.mean(nll, axis=0)

    return rval


def GaussianMSE(y, parameters):
    '''
    Compute MSE for model that outputs mean and std. 
    y = Nx1 vector of target (data) values.
    parameters = N x 2 matrix. 
    '''
    # M = 1
    mean = parameters[:, 0]
    ymean = y[:, 0]
    # sigma = parameters[:, 1] # Not used.
    rval = K.mean(K.square(mean - ymean), axis=-1)
    return rval


def MixtureNLL(y, parameters):
    '''
    Compute NLL for mixture model.
    y = Nx1 vector of target (data) values.
    parameters = N x 3M matrix of mixture model parameters.
    '''
    raise Exception('Not implemented yet...')
    M = K.shape(parameters)[-1] / 3

    means = parameters[:, 0 * M: 1 * M]
    sigmas = parameters[:, 1 * M: 2 * M]
    weights = parameters[:, 2 * M: 3 * M]

    def component_normal_likelihood(i, mus, sis, als, tr):
        mu = mus[:, i]
        si = sis[:, i]
        al = als[:, i]

        two_sigma_squared = 2 * (si ** 2)
        squared_difference = (mu - tr) ** 2
        exponent = K.exp(-squared_difference / two_sigma_squared)
        sigma_root_two_pi = si * np.sqrt(2 * np.pi)
        likelihood = exponent / sigma_root_two_pi
        return al * likelihood

    r, _ = scan(
        fn=component_normal_likelihood,
        outputs_info=None,
        sequences=[K.T.arange(M)],
        non_sequences=[means, sigmas, weights, y[:, 0]])

    return -K.log(K.sum(r, axis=0))
