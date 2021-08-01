import numpy as np

from lightspot.priors import Uniform


def test_uniform_prior_is_flat():
    rng = np.random.default_rng(42)
    prior = Uniform()
    samples = prior(rng.random((30_000, 1)))
    hist = np.histogram(samples, density=True)[0]
    assert np.max(np.abs(hist - np.ones(10))) < 0.03
