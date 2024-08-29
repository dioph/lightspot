import numpy as np
import pytest

from lightspot.priors import Dirac, Uniform
from lightspot.sampler import AbstractModel


@pytest.fixture
def linear_model():
    class LinearRegression(AbstractModel):
        def __init__(self, t, y, dy=None, priors=None):
            defaults = {"a": Uniform(-100, 100), "b": Uniform(-100, 100)}
            super(LinearRegression, self).__init__(defaults, t, y, dy, priors)

        def predict(self, t, theta):
            return np.dot(theta, np.vstack([t, np.ones_like(t)]))

    return LinearRegression


@pytest.fixture
def t():
    return np.arange(0, 30, 0.2)


@pytest.fixture
def y_linear(t):
    return 10 * t + 2


@pytest.fixture
def yerr(t):
    rng = np.random.default_rng(42)
    return np.abs(rng.standard_normal(t.size) * 30)


@pytest.fixture
def y(y_linear, yerr):
    rng = np.random.default_rng(42)
    return y_linear + rng.standard_normal(y_linear.size) * yerr


def test_prior_transform(t, y, yerr, linear_model):
    model = linear_model(t, y, yerr)
    assert np.all(model.prior_transform([0, 0]) == [-100, -100])
    assert np.all(model.prior_transform([0.5, 1.0]) == [0, 100])


def test_catch_wrong_ndim(t, y, yerr, linear_model):
    with pytest.raises(ValueError):
        _ = linear_model(t, y, yerr, priors={("a", "b"): Uniform()})
    with pytest.raises(ValueError):
        _ = linear_model(t, y, yerr, priors={"a": Uniform(ndim=2)})


def test_catch_invalid_key(t, y, yerr, linear_model):
    with pytest.raises(KeyError):
        _ = linear_model(t, y, yerr, priors={"a": Uniform(), "c": Uniform()})


def test_predict_true_params(t, y_linear, linear_model):
    model = linear_model(t, y_linear)
    assert np.allclose(model.predict(t, [10, 2]), y_linear)
    assert model.chi([[10, 2]]) == 0


def test_default_priors(t, y, yerr, linear_model):
    model = linear_model(t, y, yerr)
    assert model.priors == model.defaults


def test_param_names(t, y, yerr, linear_model):
    model = linear_model(t, y, yerr)
    assert model.param_names == ["a", "b"]


def test_fix_one_param(t, y, yerr, linear_model):
    model = linear_model(t, y, yerr, priors={"a": Dirac(10)})
    assert model.ndim == 1
    # assert model.fit_names == ["${{b}}$"]
    assert isinstance(model.id_priors[(0,)], Dirac)
    assert np.all(model.prior_transform(0.5) == [10, 0])


def test_y_follows_t_dtype(t, y, yerr, linear_model):
    assert y.dtype == "float64"
    model = linear_model(t.astype("float32"), y, yerr)
    assert model.y.dtype == "float32"
    assert model.dy.dtype == "float32"
