import unittest

import numpy as np

from lightspot.model import SpotModel, macula


class TestSpotModel(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(0, 30, 0.02)
        self.tstart = np.array([self.t[0] - .01])
        self.tend = np.array([self.t[-1] + .01])
        self.theta = np.array([1.04719755e+00,  9.00000000e+00, 0.00000000e+00,
                               0.00000000e+00,  0.00000000e+00, 7.00000000e-01,
                               0.00000000e+00,  0.00000000e+00, 0.00000000e+00,
                               7.00000000e-01,  0.00000000e+00, 0.00000000e+00,
                               1.04719755e+00, -1.57079633e+00, 5.23598776e-01,
                               5.23598776e-01,  2.09439510e-01, 1.04719755e-01,
                               2.20000000e-01,  2.20000000e-01, 0.00000000e+00,
                               0.00000000e+00,  2.00000000e+02, 2.00000000e+02,
                               5.00000000e+00,  5.00000000e+00, 5.00000000e+00,
                               5.00000000e+00,  1.00000000e+00, 1.00000000e+00])
        self.theta_star = self.theta[:12]
        self.theta_spot = self.theta[12:28].reshape(8, -1)
        self.theta_inst = self.theta[28:].reshape(2, -1)
        self.y = macula(self.t, self.theta_star, self.theta_spot, self.theta_inst,
                        self.tstart, self.tend)[0]
        self.model = SpotModel(self.t, self.y, 2)

    def test_perfect_fit_chisqr(self):
        self.assertTrue(np.allclose(self.y, self.model.predict(self.t, self.theta)))
        self.assertEqual(0, self.model.chi(self.theta))

    def test_model_ndim(self):
        self.assertEqual(self.model.ndim, self.theta.size)
        model = SpotModel(self.t, self.y, 1)
        self.assertEqual(22, model.ndim)

    def test_sample_shapes(self):
        self.assertRaises(AssertionError, self.model.sample, np.random.rand(20))
        theta = self.model.sample(np.random.rand(self.theta.size))
        self.assertEqual(self.model.ndim, theta.size)


if __name__ == '__main__':
    unittest.main()
