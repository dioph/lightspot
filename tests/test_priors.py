from lightspot.priors import Uniform
import numpy as np


def test_uniform_prior_is_flat():
    rng = np.random.default_rng(42)
    prior = Uniform()
    samples = prior.sample(*rng.random(30_000))
    hist = np.histogram(samples, density=True)[0]
    assert np.max(np.abs(hist - np.ones(10))) < 0.03
