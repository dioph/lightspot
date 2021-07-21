import os

from lightspot.model import SpotModel, macula
import numpy as np
import pytest


@pytest.fixture
def theta():
    star = np.array([1, 9, 0, 0, 0, 0.7, 0, 0, 0, 0.7, 0, 0])
    spot = np.array([1, -1.5, 0.5, 0.5, 0.2, 0.1, 0.2, 0.2, 0, 0, 200, 200, 5, 5, 5, 5])
    inst = np.array([1, 1])
    return np.hstack([star, spot, inst])


@pytest.fixture
def t():
    return np.arange(0, 30, 0.02)


@pytest.fixture
def y(t, theta):
    tstart = np.array([t[0] - 0.01])
    tend = np.array([t[-1] + 0.01])
    theta_star = theta[:12]
    theta_spot = theta[12:28].reshape(8, -1)
    theta_inst = theta[28:].reshape(2, -1)
    return macula(
        t,
        theta_star,
        theta_spot,
        theta_inst,
        tstart,
        tend,
    )[0]


@pytest.fixture
def model(t, y):
    return SpotModel(t, y, 2)


def test_perfect_fit_chisqr(model, t, y, theta):
    assert np.allclose(y, model.predict(t, theta))
    assert model.chi(theta) == 0


def test_model_ndim(model, t, y, theta):
    assert model.ndim == theta.size
    new_model = SpotModel(t, y, 1)
    assert new_model.ndim == 22


def test_sample_shapes(model, theta):
    with pytest.raises(AssertionError):
        _ = model.sample(np.random.rand(20))
    new_theta = model.sample(np.random.rand(theta.size))
    assert model.ndim == new_theta.size


def test_use_gpu_same_result(model, t, y, theta):
    if "CUDA_PATH" not in os.environ:
        pytest.skip("skipping CUDA tests")
    model_gpu = SpotModel(t, y, 2, use_gpu=True)
    assert np.allclose(y, model_gpu.predict(t, theta))
