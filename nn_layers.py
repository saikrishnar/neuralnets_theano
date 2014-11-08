__author__ = 'Siddharth Pramod'
__email__ = 'spramod1@umbc.edu'
__docformat__ = 'restructedtext en'

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from support import DispatchTable
from neuralnets_theano.nn_funcs import activation_funcs


class NNetLayer(object):
    """ A simple ANN Layer class."""
    def __init__(self, x=None, n_in=None, n_out=None, params=None, activation='relu'):
        """ :type x:  theano shared variable
            :param x: input to the neural net

            :type n_in:  int
            :param n_in: number of input units

            :type n_out:  int
            :param n_out: number of class labels

            :type params:  theano shared variable
            :param params: list/tuple containing [weights, bias] connecting input to the layer,
                           set to None if not shared

            :type activation:  str
            :param activation: activation function name: one of 'relu', 'sigmoid', 'tanh'
            """
        self.x = x
        if params:
            self.w = params[0]
            self.b = params[1]
        else:
            bound = np.sqrt(6. / (n_in + n_out))        # this bound is defined for tanh
            #TODO: bound for logistic function
            self.w = theano.shared(value=np.asarray(np.random.uniform(low=-bound, high=bound, size=(n_in, n_out)),
                                                    dtype=theano.config.floatX),
                                   name='w', borrow=True)
            self.b = theano.shared(value=np.zeros(n_out, dtype=theano.config.floatX), name='b', borrow=True)
        self.params = [self.w, self.b]
        self.activation = activation_funcs[activation]
        self.output = self.activation(T.dot(self.x, self.w) + self.b)


class DropoutLayer(NNetLayer):
    def __init__(self, x=None, n_in=None, n_out=None, params=None, activation='relu', dropout_rate=0):
        super(DropoutLayer, self).__init__(x=x, n_in=n_in, n_out=n_out, params=params, activation=activation)
        rng = np.random.RandomState()
        srng = RandomStreams(rng.randint(2 ** 30))
        retain_rate = 1 - dropout_rate
        mask = srng.binomial(n=1, p=retain_rate, size=self.output.shape)
        self.output = self.output * T.cast(mask, theano.config.floatX)      # 'Cast' is supposed hack to keep it on GPU


layer_funcs = {'vanilla': NNetLayer,
               'dropout': DropoutLayer}
layer_funcs = DispatchTable(layer_funcs)