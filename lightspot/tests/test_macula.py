import unittest

import numpy as np

from lightspot.macula import macula


class TestMacula(unittest.TestCase):
    def setUp(self):
        self.theta_star = np.array([np.pi / 3, 9, 0, 0, 0., .7, 0, 0, 0, .7, 0, 0])
        self.theta_spot = np.array([[np.pi / 3, -np.pi / 2],
                                    [np.pi / 6, np.pi / 6],
                                    [np.pi / 15, np.pi / 30],
                                    [.22, .22],
                                    [0, 0],
                                    [200, 200],
                                    [5, 5],
                                    [5, 5]])
        self.theta_inst = np.array([[1.0, 1.0],
                                    [1.0, 1.0]])

    def test_no_segfault_with_big_time_array(self):
        t = np.arange(0, 30, 0.002)  # size 15000
        _ = macula(t, self.theta_star, self.theta_spot, self.theta_inst,
                   [-0.01, 9.99], [30., 10.],
                   derivatives=True, temporal=True, tdeltav=True)

    def test_bad_args_throw_exceptions(self):
        t = np.arange(0, 30, 0.02)
        # wrong size of tstart/tend
        self.assertRaises(RuntimeError, macula,
                          t, self.theta_star, self.theta_spot, self.theta_inst,
                          [-0.01], [30.])
        # too few theta_spot rows
        self.assertRaises(RuntimeError, macula,
                          t, self.theta_star, self.theta_spot[:-1], self.theta_inst,
                          [-0.01, 9.99], [10., 30.])
        # wrong theta_star ndim
        self.assertRaises(RuntimeError, macula,
                          t, np.array([self.theta_star]), self.theta_spot, self.theta_inst,
                          [-0.01, 9.99], [10., 30.])


if __name__ == '__main__':
    unittest.main()
