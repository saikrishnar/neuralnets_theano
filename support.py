__author__ = 'Siddharth Pramod'
__email__ = 'spramod1@umbc.edu'
__docformat__ = 'restructedtext en'


def print_table(table):
    """ Pretty print a table provided as a list of rows."""
    col_size = [max(len(str(val)) for val in column) for column in zip(*table)]
    print ('=======================================================================')
    for line in table:
        print ("    ".join("{0:{1}}".format(val, col_size[i]) for i, val in enumerate(line)))
    print ('=======================================================================')


class DispatchTable(dict):
    """ A dispatch table class. Subclass of dict that implements "NotImplementedError". """
    def __init__(self, dispatchdict):
        """ Initialized with a dictionary."""
        super(DispatchTable, self).__init__(dispatchdict)

    def __getitem__(self, key):
        try:
            return super(DispatchTable, self).__getitem__(key)
        except KeyError:
            raise NotImplementedError(key)


if __name__ == '__main__':
    import theano.tensor as T

    def test_dispatchtable():
        activation_funcs = {'sigmoid': T.nnet.sigmoid,
                            'tanh': T.tanh,
                            'relu': lambda x: x * (x > 0)}
        activation_funcs = DispatchTable(activation_funcs)
        print activation_funcs['relu']
        activation_funcs['abc'] = lambda x: x + 1
        print activation_funcs['abc']
        print activation_funcs['cba']
        return None

    test_dispatchtable()