__author__ = 'Siddharth Pramod'
__email__ = 'spramod1@umbc.edu'
__docformat__ = 'restructedtext en'

import abc

import theano
import theano.tensor as T
import numpy as np

from nn_layers import layer_funcs
from nn_funcs import activation_funcs, regularization_funcs


class Classifier(object):
    """ Abstract base class for classifiers."""
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.y_pred = None

    @abc.abstractmethod
    def get_cost_updates(self, y_true=None, regularization='l1', reg_wt=0.01, learning_rate=0.1):
        return

    def get_error_rate(self, y_true):
        """ Returns the error rate.
            :type y_true:  Theano shared variable
            :param y_true: Target

            :rtype  Theano shared variable
            :return mean error
            """
        return T.mean(T.neq(self.y_pred, y_true))

    def get_accuracy(self, y_true):
        """ ~~~~~~~~~~ Untested ~~~~~~~~~~
            Returns accuracy of predictions.
            :type y_true:  Theano shared variable
            :param y_true: Target

            :rtype  Theano shared variable
            :return accuracy
            """
        return T.mean(T.eq(self.y_pred, y_true))

    def get_precision(self, y_true):
        """ ~~~~~~~~~~ Untested ~~~~~~~~~~
            Returns precision of binary predictions.
            :type y_true:  Theano shared variable
            :param y_true: Target

            :rtype  Theano shared variable
            :return precision
            """
        true_positives = T.sum(T.eq(self.y_pred, True) * T.eq(y_true, True))
        test_outcome_positives = T.sum(T.eq(self.y_pred, True))

        return true_positives / test_outcome_positives

    def get_recall(self, y_true):
        """ ~~~~~~~~~~ Untested ~~~~~~~~~~
            Returns recall/sensitivity of binary predictions.
            :type y_true:  Theano shared variable
            :param y_true: Target

            :rtype  Theano shared variable
            :return recall/sensitivity
            """
        true_positives = T.sum(T.eq(self.y_pred, True) * T.eq(y_true, True))
        condition_positives = T.sum(T.eq(y_true, True))

        return true_positives / condition_positives

    def get_specificity(self, y_true):
        """ ~~~~~~~~~~ Untested ~~~~~~~~~~
            Returns specificity of binary predictions.
            :type y_true:  Theano shared variable
            :param y_true: Target

            :rtype  Theano shared variable
            :return recall/sensitivity
            """
        true_negatives = T.sum(T.eq(self.y_pred, False) * T.eq(y_true, False))
        condition_negatives = T.sum(T.eq(y_true, False))

        return true_negatives / condition_negatives


class SoftmaxClassifier(Classifier):
    """ A softmax classifier class that can be used either as a standalone classifier or as a final layer on an ANN."""
    # TODO: Check if call to __init__ of super class is required by programming practice
    def __init__(self, x=None, config=None, params=None):
        """ :type x:  theano shared variable
            :param x: input to the Softmax layer

            :type n_in:  int
            :param n_in: number of input units

            :type n_out:  int
            :param n_out: number of class labels
            """

        n_in = config[0]
        n_out = config[1]
        self.x = x if x else T.fmatrix('x')
        if params:
            self.w = params[0]
            self.b = params[1]
        else:
            self.w = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='w', borrow=True)
            self.b = theano.shared(value=np.zeros(n_out, dtype=theano.config.floatX), name='b', borrow=True)
        self.params = [self.w, self.b]
        self.p_y_given_x = T.nnet.softmax(T.dot(x, self.w) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def get_cost_updates(self, y_true=None, regularization='l1', reg_wt=0.01, learning_rate=0.1):
        """ Implements log-loss with regularization.
            :type y_true:  Theano shared variable
            :param y_true: Target

            :type regularization:  str
            :param regularization: regularization type, default='l1', set to 'none' for no regularization

            :type reg_wt:  float
            :param reg_wt: weight for regularization term, default=0.01

            :type learning_rate:  float
            :param learning_rate: learning rate for updates

            :return log_loss + regularization term
            """
        log_loss = -T.mean(T.log(self.p_y_given_x)[T.arange(y_true.shape[0]), y_true])
        reg_term = reg_wt * regularization_funcs[regularization](self.w)
        cost = log_loss + reg_term
        gparams = T.grad(cost=cost, wrt=self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))
        return cost, updates

    def predict(self, x_test):
        predictor = theano.function(inputs=self.x, outputs=self.y_pred, givens={self.x: x_test})


class FFNN(Classifier):
    """ Class for vanilla feedforward neural nets."""
    def __init__(self, x=None, config=None, activation='relu', layer_type='vanilla'):
        """ :type x:  theano shared variable
            :param x: input to the neural net

            :type config:  list
            :param config: list of layer sizes beginning with n_in (input size), ending with n_out(number of classes)

            :type activation:  str
            :param activation: activation function name: one of 'relu', 'sigmoid', 'tanh'

            :type Layer:  Class object
            :param Layer: Neural Net Layer class, leave as default.
                          Typically used by inheriting classes that use alternate Layer classes.
            """
        self.x = x if x else T.fmatrix('x')
        self.activation = activation_funcs[activation]
        self.nnet = []
        self.params = []
        inp = self.x
        Layer_class = layer_funcs[layer_type]
        for i in range(1, len(config) - 1):
            layer = Layer_class(x=inp, n_in=config[i - 1], n_out=config[i], params=None, activation=activation)
            self.nnet.append(layer)
            self.params.extend(layer.params)
            inp = layer.output

        self.classifier = SoftmaxClassifier(x=self.nnet[-1].output, config=[config[-2], config[-1]])
        self.params.extend(self.classifier.params)
        self.get_error_rate = self.classifier.get_error_rate

    def get_cost_updates(self, y_true=None, learning_rate=0.01, regularization='l1', reg_wt=0.01):
        """ Implements log-loss with regularization.
            :type y_true:  Theano shared variable
            :param y_true: Target

            :type regularization:  str
            :param regularization: regularization type, default='l1', set to 'none' for no regularization

            :type reg_wt:  float
            :param reg_wt: weight for regularization term, default=0.01

            :type learning_rate:  float
            :param learning_rate: learnLayer_class(x=inp, n_in=config[i - 1], n_out=config[i], params=None, activation=activation)ing rate for updates

            :return log_loss + regularization term
            """
        log_loss = -T.mean(T.log(self.classifier.p_y_given_x)[T.arange(y_true.shape[0]), y_true])
        reg_func = regularization_funcs[regularization]
        reg_term = (sum([reg_func(layer.w) for layer in self.nnet])
                    + reg_func(self.classifier.w)) * reg_wt/(len(self.nnet) + 1)
        cost = log_loss + reg_term
        gparams = T.grad(cost=cost, wrt=self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))
        return cost, updates


class FFNN_Dropout(FFNN):
    """ Class that implements a Feed Forward Neural Network with dropout in every layer."""
    def __init__(self, x=None, config=None, activation='relu', dropout_rate=0):
        super(FFNN_Dropout, self).__init__(x=x, config=config, activation=activation, layer_type='dropout')
        # Following creates a duplicate neural net that does not use dropout. Use for testing.
        # TODO: make more efficient
        retain_rate = 1 - dropout_rate
        Layer_class = layer_funcs['vanilla']
        self.nnet_sans_dropout = []
        inp = self.x
        for i in range(len(self.nnet)):
            params = (self.nnet[i].w * retain_rate, self.nnet[i].b)
            layer = Layer_class(x=inp, n_in=config[i], n_out=config[i + 1], params=params, activation=activation)
            self.nnet_sans_dropout.append(layer)
            inp = layer.output
        classifier_params = (self.classifier.w, self.classifier.b)
        self.classifier_sans_dropout = SoftmaxClassifier(x=self.nnet_sans_dropout[-1].output,
                                                         config=[config[-2], config[-1]], params=classifier_params)
        self.get_error_rate = self.classifier_sans_dropout.get_error_rate
        self.get_accuracy = self.classifier_sans_dropout.get_accuracy
        self.get_precision = self.classifier_sans_dropout.get_precision
        self.get_recall = self.classifier_sans_dropout.get_recall