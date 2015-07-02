import os
import numpy as np
from bolero.representation import DMPBehavior
from nose.tools import assert_equal, assert_raises_regexp
from numpy.testing import assert_array_equal, assert_array_almost_equal


CURRENT_PATH = os.sep.join(__file__.split(os.sep)[:-1])
DMP_CONFIG_FILE = CURRENT_PATH + os.sep + "dmp_model.yaml"
if not CURRENT_PATH:
    DMP_CONFIG_FILE = "dmp_model.yaml"

n_task_dims = 1


def eval_loop(beh, xva):
    beh.set_inputs(xva)
    beh.step()
    beh.get_outputs(xva)

def test_dimensions_do_not_match():
    beh = DMPBehavior()
    assert_raises_regexp(ValueError, "Input and output dimensions must match",
                         beh.init, 1, 2)

def test_default_dmp():
    beh = DMPBehavior()
    beh.init(3 * n_task_dims, 3 * n_task_dims)

    assert_equal(beh.get_n_params(), 50 * n_task_dims)
    assert_array_equal(beh.get_params(), np.zeros(50 * n_task_dims))

    xva = np.zeros(3 * n_task_dims)
    beh.reset()
    t = 0
    while beh.can_step():
        eval_loop(beh, xva)
        t += 1
    assert_equal(t, 101)
    assert_array_equal(xva[:n_task_dims], np.zeros(n_task_dims))
    assert_array_equal(xva[n_task_dims:-n_task_dims], np.zeros(n_task_dims))
    assert_array_equal(xva[-n_task_dims:], np.zeros(n_task_dims))


def test_dmp_from_config():
    beh = DMPBehavior(DMP_CONFIG_FILE)
    beh.init(18, 18)

    xva = np.zeros(18)
    beh.reset()
    while beh.can_step():
        eval_loop(beh, xva)


def test_metaparameter_not_permitted():
    beh = DMPBehavior()
    beh.init(3, 3)
    assert_raises_regexp(ValueError, "Meta parameter .* is not allowed",
                         beh.set_meta_parameters, ["unknown"], [None])


def test_change_goal():
    beh = DMPBehavior()
    beh.init(3 * n_task_dims, 3 * n_task_dims)

    beh.set_meta_parameters(["g"], [np.ones(n_task_dims)])
    xva = np.zeros(3 * n_task_dims)
    beh.reset()
    while beh.can_step():
        eval_loop(beh, xva)
    assert_array_almost_equal(xva[:n_task_dims], np.ones(n_task_dims),
                              decimal=3)
    assert_array_almost_equal(xva[n_task_dims:-n_task_dims],
                              np.zeros(n_task_dims), decimal=2)
    assert_array_almost_equal(xva[-n_task_dims:], np.zeros(n_task_dims),
                              decimal=1)


def test_change_goal_velocity():
    beh = DMPBehavior()
    beh.init(3 * n_task_dims, 3 * n_task_dims)

    beh.set_meta_parameters(["gd"], [np.ones(n_task_dims)])
    xva = np.zeros(3 * n_task_dims)
    beh.reset()
    while beh.can_step():
        eval_loop(beh, xva)
    assert_array_almost_equal(xva[:n_task_dims], np.zeros(n_task_dims),
                              decimal=3)
    assert_array_almost_equal(xva[n_task_dims:-n_task_dims],
                              np.ones(n_task_dims), decimal=2)
    assert_array_almost_equal(xva[-n_task_dims:], np.zeros(n_task_dims),
                              decimal=1)


def test_change_execution_time():
    beh = DMPBehavior()
    beh.init(3 * n_task_dims, 3 * n_task_dims)

    beh.set_meta_parameters(["execution_time"], [2.0])
    xva = np.zeros(3 * n_task_dims)
    beh.reset()
    t = 0
    while beh.can_step():
        eval_loop(beh, xva)
        t += 1
    assert_equal(t, 201)


def test_change_weights():
    beh = DMPBehavior()
    beh.init(3 * n_task_dims, 3 * n_task_dims)

    beh.set_params(np.ones(50 * n_task_dims))
    xva = np.zeros(3 * n_task_dims)
    beh.reset()
    while beh.can_step():
        eval_loop(beh, xva)
    assert_array_almost_equal(xva[:n_task_dims], np.zeros(n_task_dims),
                              decimal=3)
    assert_array_almost_equal(xva[n_task_dims:-n_task_dims],
                              np.zeros(n_task_dims), decimal=2)
    assert_array_almost_equal(xva[-n_task_dims:], np.zeros(n_task_dims),
                              decimal=1)

def test_more_steps_than_allowed():
    beh = DMPBehavior()
    beh.init(3 * n_task_dims, 3 * n_task_dims)

    xva = np.zeros(3 * n_task_dims)
    beh.reset()
    while beh.can_step():
        eval_loop(beh, xva)

    last_x = xva[:n_task_dims].copy()

    eval_loop(beh, xva)

    assert_array_equal(xva[:n_task_dims], last_x)
    assert_array_equal(xva[n_task_dims:-n_task_dims], np.zeros(n_task_dims))
    assert_array_equal(xva[-n_task_dims:], np.zeros(n_task_dims))
