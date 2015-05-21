import numpy as np
from nose.tools import assert_less, assert_equal
from sklearn.utils.testing import assert_warns
from numpy.testing import assert_array_almost_equal
from bolero.optimizer import CMAESOptimizer, fmin


def test_cmaes_no_initial_params():
    opt = CMAESOptimizer()
    opt.init(10)
    params = np.empty(10)
    opt.get_next_parameters(params)


def test_cmaes_diagonal_cov():
    opt = CMAESOptimizer(covariance=np.zeros(10))
    opt.init(10)
    params = np.empty(10)
    opt.get_next_parameters(params)


def test_cmaes_minimize():
    _, f = fmin(lambda x: np.linalg.norm(x), cma_type="standard",
                x0=np.zeros(2), random_state=0, maxfun=300)
    assert_less(f, 1e-5)


def test_ipop_cmaes():
    _, f = fmin(lambda x: np.linalg.norm(x), cma_type="ipop",
                x0=np.zeros(2), random_state=0, maxfun=300)
    assert_less(f, 1e-5)


def test_bipop_cmaes():
    _, f = fmin(lambda x: np.linalg.norm(x), cma_type="bipop",
                x0=np.zeros(2), random_state=0, maxfun=300)
    assert_less(f, 1e-5)


def test_cmaes_get_best_params_mean():
    opt = CMAESOptimizer()
    opt.init(10)
    params = np.empty(10)
    opt.get_next_parameters(params)
    opt.set_evaluation_feedback(np.array([0.0]))
    best_params = opt.get_best_parameters(method="mean")
    assert_array_almost_equal(np.zeros(10), best_params)


def test_cmaes_get_best_params_best():
    opt = CMAESOptimizer()
    opt.init(10)
    params = np.empty(10)
    opt.get_next_parameters(params)
    opt.set_evaluation_feedback(np.array([0.0]))
    best_params = opt.get_best_parameters(method="best")
    assert_array_almost_equal(params, best_params)
