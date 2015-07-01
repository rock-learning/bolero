import numpy as np
from bolero.datasets import make_minimum_jerk
from dmp import DMP, imitate_dmp
from numpy.testing import assert_array_almost_equal


x0 = np.zeros(1)
g = np.ones(1)
dt = 0.001
execution_time = 1.0
n_features = 10


def test_imitation():
    X, Xd, Xdd = make_minimum_jerk(x0, g, execution_time, dt)
    dmp = DMP(execution_time=execution_time, dt=dt, n_features=n_features)
    imitate_dmp(dmp, X, Xd, Xdd, set_weights=True)
    Y, Yd, Ydd = episode(dmp, x0, g)
    assert_array_almost_equal(X[:, :, 0], Y, decimal=4)
    assert_array_almost_equal(Xd[:, :, 0], Yd, decimal=2)
    assert_array_almost_equal(Xdd[:, :, 0], Ydd, decimal=1)


def test_temporal_scaling():
    X, Xd, Xdd = make_minimum_jerk(x0, g, execution_time, dt)
    dmp = DMP(execution_time=execution_time, dt=dt, n_features=n_features)
    imitate_dmp(dmp, X, Xd, Xdd, set_weights=True)
    Y = episode(dmp, x0, g)[0]

    scaled_execution_time = 0.5
    dmp.set_metaparameters(["execution_time"], [scaled_execution_time])
    dmp.reset()
    Yr = episode(dmp, x0, g)[0]

    assert_array_almost_equal(Y[0, ::2], Yr[0], decimal=4)


def episode(dmp, x0, g):
    x = x0.copy()
    xd = np.zeros_like(x)
    xdd = np.zeros_like(x)

    X, Xd, Xdd = [], [], []

    while dmp.can_step():
        x, xd, xdd = dmp.execute_step(x, xd, xdd, x0=x0, g=g)
        X.append(x.copy())
        Xd.append(xd.copy())
        Xdd.append(xdd.copy())

    return np.array(X).T, np.array(Xd).T, np.array(Xdd).T
