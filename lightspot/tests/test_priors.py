import unittest

import numpy as np

from lightspot.priors import Uniform


class TestPrior(unittest.TestCase):
    def test_uniform(self):
        np.random.seed(42)
        prior = Uniform()
        samples = prior.sample(*np.random.random(30_000))
        hist = np.histogram(samples, density=True)[0]
        self.assertLess(np.max(np.abs(hist - np.ones(10))), 0.03)


if __name__ == '__main__':
    unittest.main()
