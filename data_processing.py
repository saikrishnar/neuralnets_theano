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


if __name__ == '__main__':
    import unittest

    class TestPreprocessor(unittest.TestCase):
        def setUp(self):
            self.a = np.random.rand(10, 4)
            self.preprocessor = Preprocessor()

        def test_set_mean(self):
            self.preprocessor.set_mean(self.a)
            np.testing.assert_almost_equal(self.preprocessor.mean, np.mean(self.a, axis=0))


        def test_set_variance(self):
            self.preprocessor.set_variance(self.a)
            np.testing.assert_almost_equal(self.preprocessor.variance, np.var(self.a, axis=0))

        def test_apply_mean(self):
            self.preprocessor.set_mean(self.a)
            self.a_meaned = self.preprocessor.apply_mean(self.a)
            np.testing.assert_almost_equal(self.a, self.a_meaned + self.preprocessor.mean)

        def test_apply_variance(self):
            self.preprocessor.set_variance(self.a)
            self.a_vared = self.preprocessor.apply_variance(self.a)
            np.testing.assert_almost_equal(self.a, self.a_vared * self.preprocessor.variance)

        def tearDown(self):
            pass

    unittest.main()