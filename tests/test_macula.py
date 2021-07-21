from lightspot.macula import macula
import numpy as np
import pytest


@pytest.fixture
def theta_star():
    return np.array([np.pi / 3, 9, 0, 0, 0.0, 0.7, 0, 0, 0, 0.7, 0, 0])


@pytest.fixture
def theta_spot():
    return np.array(
        [
            [np.pi / 3, -np.pi / 2],
            [np.pi / 6, np.pi / 6],
            [np.pi / 15, np.pi / 30],
            [0.22, 0.22],
            [0, 0],
            [200, 200],
            [5, 5],
            [5, 5],
        ]
    )


@pytest.fixture
def theta_inst():
    return np.array([[1.0, 1.0], [1.0, 1.0]])


def test_no_segfault_with_big_time_array(theta_star, theta_spot, theta_inst):
    t = np.arange(0, 30, 0.002)  # size 15000
    _ = macula(
        t,
        theta_star,
        theta_spot,
        theta_inst,
        [-0.01, 9.99],
        [30.0, 10.0],
        derivatives=True,
        temporal=True,
        tdeltav=True,
    )


def test_bad_args_throw_exceptions(theta_star, theta_spot, theta_inst):
    t = np.arange(0, 30, 0.02)
    # wrong size of tstart/tend
    with pytest.raises(RuntimeError):
        _ = macula(t, theta_star, theta_spot, theta_inst, [-0.01], [30.0])
    # too few theta_spot rows
    with pytest.raises(RuntimeError):
        _ = macula(
            t, theta_star, theta_spot[:-1], theta_inst, [-0.01, 9.99], [10.0, 30.0]
        )
    # wrong theta_star ndim
    with pytest.raises(RuntimeError):
        _ = macula(
            t,
            np.array([theta_star]),
            theta_spot,
            theta_inst,
            [-0.01, 9.99],
            [10.0, 30.0],
        )
