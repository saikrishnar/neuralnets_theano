__author__ = 'Siddharth Pramod'
__email__ = 'spramod1@umbc.edu'
__docformat__ = 'restructedtext en'


import numpy as np


class Preprocessor(object):
    def __init__(self, mean=None, variance=None):
        self.mean = mean
        self.variance = variance

    def set_mean(self, numpy_2darray, axis=0):
        self.mean = np.mean(numpy_2darray, axis=axis)

    def set_variance(self, numpy_2darray, axis=0):
        self.variance = np.var(numpy_2darray, axis=axis)

    def apply_mean(self, numpy_2darray):
        numpy_2darray = numpy_2darray - self.mean
        return numpy_2darray

    def apply_variance(self, numpy_2darray):
        numpy_2darray = numpy_2darray / self.variance
        return numpy_2darray

    def set_apply_mean(self, numpy_2darray, axis=0):
        self.mean = np.mean(numpy_2darray, axis=axis)
        numpy_2darray = numpy_2darray - self.mean
        return numpy_2darray

    def set_apply_variance(self, numpy_2darray, axis=0):
        self.variance = np.var(numpy_2darray, axis=axis)
        numpy_2darray = numpy_2darray / self.variance
        return numpy_2darray