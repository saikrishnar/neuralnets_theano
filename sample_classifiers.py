__author__ = 'Siddharth Pramod'
__email__ = 'spramod1@umbc.edu'
__docformat__ = 'restructedtext en'

import numpy as np
from sklearn import datasets

from nn_classifiers import SoftmaxClassifier, FFNN, FFNN_Dropout
from neuralnets_theano.optimizations import sgd


def test_softmax(x, y):
    learning_rate = 0.1
    train_validation_split = 0.7
    Classifier = SoftmaxClassifier
    classifier_options = dict(config=[])
    best_error, best_iter = sgd(Classifier=Classifier, classifier_options=classifier_options, x_data=x, y_data=y,
                                train_validation_split=train_validation_split, learning_rate=learning_rate,
                                max_iter=500)
    return None


def test_ffnn(x, y):
    learning_rate = 0.1
    train_validation_split = 0.7
    Classifier = FFNN
    config = [10, 10]
    classifier_options = dict(config=config)
    best_error, best_iter = sgd(Classifier=Classifier, classifier_options=classifier_options, x_data=x, y_data=y,
                                train_validation_split=train_validation_split, learning_rate=learning_rate,
                                max_iter=500)
    return None


def test_dropoutnn(x, y):
    learning_rate = 0.1
    train_validation_split = 0.7
    Classifier = FFNN_Dropout
    config = [10, 10]
    dropout_rate = 0.15
    classifier_options = dict(config=config, dropout_rate=dropout_rate)
    best_error, best_iter = sgd(Classifier=Classifier, classifier_options=classifier_options, x_data=x, y_data=y,
                                train_validation_split=train_validation_split, learning_rate=learning_rate,
                                max_iter=500)
    return None


if __name__ == '__main__':
    data = datasets.load_iris()
    shuffle_idx = np.random.permutation(len(data.data))
    x = data.data[shuffle_idx]
    y = data.target[shuffle_idx]
    # xy = np.concatenate((data.data, data.target[np.newaxis].T), axis=1)
    test_softmax(x, y)
    raw_input('HIT ENTER TO CONTINUE')
    test_ffnn(x, y)
    raw_input('HIT ENTER TO CONTINUE')
    test_dropoutnn(x, y)