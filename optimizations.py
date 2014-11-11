__author__ = 'Siddharth Pramod'
__email__ = 'spramod1@umbc.edu'
__docformat__ = 'restructedtext en'


import numpy as np
from copy import deepcopy
from collections import deque
import theano
import theano.tensor as T


def sgd_loop(classifier, training_function, training_input, validation_function, validation_input,
             max_iter=1000, tolerance=0.05, n_history=50, verbose=True):
    best_val_error = np.inf
    best_iter = 0
    # best_classifier = deepcopy(classifier)
    best_parameters = classifier.get_parameters()
    prev_n_val_error = deque([best_val_error for h in range(n_history)])
    i = 0
    done = False
    if verbose:
        print ('Iter # \t\t\t TrainCost \t\t\t ValErr \t\t\t\t PrevBestValErr')
    while (i < max_iter) and (not done):
        # TODO implement 'done' condition, aka early stopping
        i += 1
        train_cost, train_error = training_function(training_input)
        val_error = validation_function(validation_input)
        if verbose:
            print('{0} \t\t\t {1} \t\t\t {2} \t\t\t {3}'.format(i, train_cost, val_error, best_val_error))
        if val_error < best_val_error:
            best_val_error = deepcopy(val_error)
            best_iter = deepcopy(i)
            # best_classifier = deepcopy(classifier)
            best_parameters = classifier.get_parameters()
            prev_n_val_error = deque([best_val_error for h in range(n_history)])
        prev_n_val_error.popleft()
        prev_n_val_error.append(val_error)
        if np.mean(np.asarray(prev_n_val_error)) > (best_val_error + tolerance):
            done = True
            if verbose:
                print('------')
                print(' Validation Error hasn\'t budged in {0} iterations, stopping sgd'.format(n_history))

    if verbose:
        print('------------------------------------------')
        print('Best validation error: {0} \n At iteration: {1}'.format(best_val_error, best_iter))

    classifier.set_parameters(best_parameters)
    # return best_val_error, best_iter, best_classifier
    return best_val_error, best_iter, classifier


def sgd(Classifier, classifier_options, x_data, y_data, train_validation_split=0.7, learning_rate=0.1, max_iter=1000):
    """ A function that takes as input the Classifier class and x, y data as numpy 2darrays and performs sgd.
        :param Classifier: classifier Class object
        :param classifier_options: options (hyper parameters) for the classifier
        :param x_data: numpy matrix containing train + validation data
        :param y_data: numpy matrix containing labels
        :param train_validation_split: the fraction of instances to use as training (rest used for validation)
        :param learning_rate: learning rate to use for updates
    """
    x_symbolic = T.fmatrix(name='x_symbolic')
    y_symbolic = T.lvector(name='y_symbolic')
    index_symbolic = T.lscalar(name='index_symbolic')

    x_data = theano.shared(np.asarray(x_data, dtype='float32'))
    y_data = theano.shared(np.asarray(y_data, dtype='int64'))

    num_instances = x_data.get_value(borrow=True).shape[0]
    idx = int(num_instances * train_validation_split)

    input_dimension = x_data.get_value(borrow=True).shape[1]
    num_classes = len(set(y_data.get_value()))
    classifier_options['config'] = [input_dimension] + classifier_options['config'] + [num_classes]
    classifier = Classifier(x=x_symbolic, **classifier_options)
    cost, updates = classifier.get_cost_updates(y_true=y_symbolic, learning_rate=learning_rate)
    error = classifier.get_error_rate(y_symbolic)

    training_function = theano.function(inputs=[index_symbolic], outputs=[cost, error], updates=updates,
                                        givens={x_symbolic: x_data[:index_symbolic],
                                                y_symbolic: y_data[:index_symbolic]})
    validation_function = theano.function(inputs=[index_symbolic], outputs=error,
                                          givens={x_symbolic: x_data[index_symbolic:],
                                                  y_symbolic: y_data[index_symbolic:]})

    best_val_error, best_iter, best_classifier = sgd_loop(classifier, training_function=training_function,
                                                          validation_function=validation_function, training_input=idx,
                                                          validation_input=idx, max_iter=max_iter, verbose=False)

    return best_val_error, best_iter, best_classifier