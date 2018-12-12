import numpy as np
from nose.tools import (assert_less, assert_greater, assert_equal,
                        assert_raises_regexp)
from sklearn.utils.testing import assert_warns
from numpy.testing import assert_array_almost_equal
from cem import CEMOptimizer


def test_cmaes_no_initial_params():
    opt = CEMOptimizer()
    opt.init(10)
    params = np.empty(10)
    opt.get_next_parameters(params)


def test_cmaes_dimensions_mismatch():
    opt = CEMOptimizer(initial_params=np.zeros(5))
    assert_raises_regexp(ValueError, "Number of dimensions", opt.init, 10)


def test_cmaes_get_best_params_mean():
    opt = CEMOptimizer()
    opt.init(10)
    params = np.empty(10)
    opt.get_next_parameters(params)
    opt.set_evaluation_feedback(np.array([0.0]))
    best_params = opt.get_best_parameters(method="mean")
    assert_array_almost_equal(np.zeros(10), best_params)


def test_cmaes_get_best_params_best():
    opt = CEMOptimizer()
    opt.init(10)
    params = np.empty(10)
    opt.get_next_parameters(params)
    opt.set_evaluation_feedback(np.array([0.0]))
    best_params = opt.get_best_parameters(method="best")
    assert_array_almost_equal(params, best_params)
