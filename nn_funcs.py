__author__ = 'Siddharth Pramod, Karan K. Budhraja'
__email__ = 'spramod1@umbc.edu, karanb1@umbc.edu'
__docformat__ = 'restructedtext en'

import theano.tensor as T

from neuralnets_theano.support import DispatchTable


# TODO: maxout
activation_funcs = {'sigmoid': T.nnet.sigmoid,
                    'tanh': T.tanh,
                    'relu': lambda x: x * (x > 0),
                    'cappedRelu': lambda x, cap: T.minimum(x * (x > 0), cap)}
activation_funcs = DispatchTable(activation_funcs)

# TODO: capped regularization
regularization_funcs = {'l1': lambda x: T.mean(abs(x)),
                        'l2': lambda x: T.mean(x ** 2),
                        'none': lambda x: 0}
regularization_funcs = DispatchTable(regularization_funcs)
