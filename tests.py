import unittest
import numpy as np

from data_processing import Preprocessor


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.a = np.random.rand(10, 4)
        self.preprocessor = Preprocessor()
        
    def tearDown(self):
        pass

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


if __name__ == '__main__':
    unittest.main()