import numpy as np

from lightspot.priors import Dirac, Normal, QuadraticLD, TruncNormal, Uniform


def test_uniform_prior_is_flat():
    rng = np.random.default_rng(42)
    prior = Uniform()
    samples = prior(rng.random((30_000, 1)))
    hist = np.histogram(samples, density=True)[0]
    assert np.max(np.abs(hist - np.ones(10))) < 0.03


def test_dirac_prior():
    prior = Dirac(42)
    assert prior.n_inputs == 0
    assert prior.n_outputs == 1
    assert prior(None) == 42


def test_normal_prior():
    rng = np.random.default_rng(42)
    prior = Normal(mu=0, sd=1)
    samples = prior(rng.random((10000, 1)))
    assert np.abs(samples.mean()) < 0.05
    assert np.abs(samples.std() - 1) < 0.05


def test_truncnormal_prior():
    rng = np.random.default_rng(42)
    prior = TruncNormal(mu=0, sd=1, xmin=-1, xmax=1)
    samples = prior(rng.random((10000, 1)))
    assert np.all(samples >= -1)
    assert np.all(samples <= 1)


def test_quadratic_ld():
    rng = np.random.default_rng(42)
    prior = QuadraticLD()
    c = prior(rng.random((100, 2)))  # 2D input -> 4D output
    assert c.shape == (100, 4)
