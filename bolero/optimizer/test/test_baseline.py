import numpy as np
from bolero.optimizer import NoOptimizer, RandomOptimizer
from bolero.utils.testing import assert_raise_message
from nose.tools import assert_false
from numpy.testing import assert_array_equal


def test_no_optimizer():
    initial_params = np.zeros(3)
    opt = NoOptimizer(initial_params)

    assert_raise_message(ValueError, "Number of dimensions", opt.init, 2)

    opt.init(3)
    params1 = np.empty(3)
    opt.get_next_parameters(params1)
    opt.set_evaluation_feedback(np.array([0.0]))
    params2 = np.empty(3)
    opt.get_next_parameters(params2)
    opt.set_evaluation_feedback(np.array([0.0]))
    assert_array_equal(params1, params2)


def test_random_optimizer():
    initial_params = np.zeros(3)
    opt = RandomOptimizer(initial_params)

    assert_raise_message(ValueError, "Number of dimensions", opt.init, 2)

    opt.init(3)
    params1 = np.empty(3)
    opt.get_next_parameters(params1)
    opt.set_evaluation_feedback(np.array([0.0]))
    params2 = np.empty(3)
    opt.get_next_parameters(params2)
    opt.set_evaluation_feedback(np.array([0.0]))
    assert_false(np.all(params1 == params2))

    opt = RandomOptimizer(initial_params, covariance=np.eye(3))
    opt.init(3)
