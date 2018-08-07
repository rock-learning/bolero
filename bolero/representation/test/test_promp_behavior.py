import numpy as np
try:
    from bolero.representation import ProMPBehavior
except ImportError:
    from nose import SkipTest
from bolero.datasets import make_minimum_jerk
from nose.tools import (assert_true, assert_raises_regexp, assert_not_equal,
                        assert_equal)
from numpy.testing import assert_array_equal, assert_array_almost_equal


n_task_dims = 2


def eval_loop(beh, xv):
    beh.set_inputs(xv)
    beh.step()
    beh.get_outputs(xv)


def test_smoke():
    beh = ProMPBehavior()
    beh.init(2, 2)
    beh.reset()
    assert_true(beh.can_step())
    beh.set_inputs(np.zeros(2))
    assert_raises_regexp(Exception, "weights", beh.step)
    params = np.zeros(beh.get_n_params())
    beh.set_params(params)
    beh.step()
    outputs = np.ones(2)
    beh.get_outputs(outputs)
    assert_not_equal(outputs[0], 1.0)
    assert_not_equal(outputs[1], 1.0)


def test_dmp_default_promp():
    beh = ProMPBehavior()
    beh.init(2 * n_task_dims, 2 * n_task_dims)
    beh.set_params(np.zeros(beh.get_n_params()))

    xv = np.zeros(2 * n_task_dims)
    beh.reset()
    t = 0
    while beh.can_step():
        eval_loop(beh, xv)
        t += 1

    assert_equal(t, 101)
    assert_array_equal(xv[:n_task_dims], np.zeros(n_task_dims))
    assert_array_equal(xv[n_task_dims:], np.zeros(n_task_dims))


def test_promp_get_set_params():
    beh = ProMPBehavior()
    beh.init(2 * n_task_dims, 2 * n_task_dims)

    assert_equal(beh.get_n_params(), 50 * n_task_dims)
    params = beh.get_params()
    assert_array_equal(params, np.zeros(50 * n_task_dims))

    random_state = np.random.RandomState(0)
    expected_params = random_state.randn(50 * n_task_dims)
    beh.set_params(expected_params)

    actual_params = beh.get_params()
    assert_array_equal(actual_params, expected_params)


def test_promp_constructor_args():
    beh = ProMPBehavior(execution_time=2)
    beh.init(2 * n_task_dims, 2 * n_task_dims)
    beh.set_params(np.zeros(beh.get_n_params()))

    xv = np.zeros(2 * n_task_dims)
    beh.reset()
    t = 0
    while beh.can_step():
        eval_loop(beh, xv)
        t += 1

    assert_equal(t, 201)


def test_promp_imitate():
    x0, g, execution_time, dt = np.zeros(2), np.ones(2), 1.0, 0.001

    beh = ProMPBehavior(execution_time, dt, 20)
    beh.init(4, 4)

    X_demo = make_minimum_jerk(x0, g, execution_time, dt)[0]

    beh.imitate(X_demo)
    X = beh.trajectory()[0]
    assert_array_almost_equal(X_demo.T[0], X, decimal=2)

    # Self-imitation
    beh.imitate(X.T[:, :, np.newaxis])
    X2 = beh.trajectory()[0]
    assert_array_almost_equal(X2, X, decimal=3)