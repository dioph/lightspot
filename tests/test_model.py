import numpy as np
import pytest

from lightspot.sampler import SpotModel

try:
    from numba import cuda

    from lightspot.gsampler import GPUSpotModel

    CUDA_AVAILABLE = True
except ModuleNotFoundError:
    CUDA_AVAILABLE = False


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
    return SpotModel(t, t, 2).predict(t, theta)[0]


@pytest.fixture
def model(t, y):
    return SpotModel(t, y, 2)


def test_perfect_fit_chisqr(model, t, y, theta):
    assert np.allclose(y, model.predict(t, theta)[0])
    assert model.chi(theta) == 0


def test_model_ndim(model, t, y, theta):
    assert model.jmax == theta.size
    new_model = SpotModel(t, y, 1)
    assert new_model.ndim == 18


def test_sample_shapes(model, theta):
    with pytest.raises(ValueError):
        _ = model.prior_transform(np.random.rand(20))
    new_theta = model.prior_transform(np.random.rand(model.ndim))
    assert model.jmax == new_theta.size


def test_fit_names(model):
    assert len(model.fit_names) == model.ndim


def test_use_gpu_same_result(t, y, theta):
    if not CUDA_AVAILABLE or not cuda.is_available():
        pytest.skip("skipping CUDA tests")
    model_gpu = GPUSpotModel(t, y, 2)
    assert np.allclose(y, model_gpu.predict(t, theta)[0].get())
