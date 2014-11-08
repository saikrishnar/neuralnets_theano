__author__ = 'Siddharth Pramod'
__email__ = 'spramod1@umbc.edu'
__docformat__ = 'restructedtext en'

import numpy as np
from sklearn import datasets
import theano
import theano.tensor as T

from nn_classifiers import SoftmaxClassifier, FFNN, FFNN_Dropout
from neuralnets_theano.optimizations import sgd_loop


def test_softmax(x, y):
    lr = 0.1
    x_in = T.fmatrix(name='x_in')
    y_tr = T.lvector(name='y_tr')
    index = T.lscalar(name='index')
    num_ins = x.get_value(borrow=True).shape[0]
    idx = int(num_ins * 0.7)
    n_in = x.get_value(borrow=True).shape[1]
    n_out = len(set(y.get_value()))
    clf = SoftmaxClassifier(x=x_in, n_in=n_in, n_out=n_out)
    cost, updates = clf.get_cost_updates(y_true=y_tr, learning_rate=lr)
    val_err = clf.get_error_rate(y_tr)
    train_func = theano.function(inputs=[index], outputs=cost, updates=updates,
                                 givens={x_in: x[:index], y_tr: y[:index]})
    val_func = theano.function(inputs=[index], outputs=val_err,
                               givens={x_in: x[index:], y_tr: y[index:]})
    sgd_loop(training_function=train_func, validation_function=val_func,
             training_input=idx, validation_input=idx)
    return None


def test_ffnn(x, y):
    lr = 0.1
    x_in = T.fmatrix(name='x_in')
    y_tr = T.lvector(name='y_tr')
    index = T.lscalar(name='index')
    num_ins = x.get_value(borrow=True).shape[0]
    idx = int(num_ins * 0.7)
    n_in = x.get_value(borrow=True).shape[1]
    n_out = len(set(y.get_value()))
    config = [n_in, 10, 10, n_out]
    clf = FFNN(x=x_in, config=config)
    cost, updates = clf.get_cost_updates(y_true=y_tr, learning_rate=lr)
    val_err = clf.get_error_rate(y_tr)
    train_func = theano.function(inputs=[index], outputs=cost, updates=updates,
                                 givens={x_in: x[:index], y_tr: y[:index]})
    val_func = theano.function(inputs=[index], outputs=val_err,
                               givens={x_in: x[index:], y_tr: y[index:]})
    sgd_loop(training_function=train_func, validation_function=val_func,
        training_input=idx, validation_input=idx)
    return None


def test_dropoutnn(x, y):
    lr = 0.1
    x_in = T.fmatrix(name='x_in')
    y_tr = T.lvector(name='y_tr')
    index = T.lscalar(name='index')
    num_ins = x.get_value(borrow=True).shape[0]
    idx = int(num_ins * 0.7)
    n_in = x.get_value(borrow=True).shape[1]
    n_out = len(set(y.get_value()))
    config = [n_in, 10, 10, n_out]
    clf = FFNN_Dropout(x=x_in, config=config, dropout_rate=0)
    cost, updates = clf.get_cost_updates(y_true=y_tr, learning_rate=lr, regularization='none')
    val_err = clf.get_error_rate(y_tr)
    train_func = theano.function(inputs=[index], outputs=cost, updates=updates,
                                 givens={x_in: x[:index], y_tr: y[:index]})
    val_func = theano.function(inputs=[index], outputs=val_err,
                               givens={x_in: x[index:], y_tr: y[index:]})
    sgd_loop(training_function=train_func, validation_function=val_func,
        training_input=idx, validation_input=idx)
    return None


if __name__ == '__main__':
    data = datasets.load_iris()
    shuffle_idx = np.random.permutation(len(data.data))
    x = data.data[shuffle_idx]
    y = data.target[shuffle_idx]
    # xy = np.concatenate((data.data, data.target[np.newaxis].T), axis=1)
    test_softmax(theano.shared(np.asarray(x, dtype='float32')),
                   theano.shared(np.asarray(y, dtype='int64')))
    raw_input('HIT ENTER TO CONTINUE')
    test_ffnn(theano.shared(np.asarray(x, dtype='float32')),
                   theano.shared(np.asarray(y, dtype='int64')))
    raw_input('HIT ENTER TO CONTINUE')
    test_dropoutnn(theano.shared(np.asarray(x, dtype='float32')),
                   theano.shared(np.asarray(y, dtype='int64')))