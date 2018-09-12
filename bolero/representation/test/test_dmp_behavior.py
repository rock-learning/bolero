import os
import numpy as np
try:
    from bolero.representation import DMPBehavior
except ImportError:
    from nose import SkipTest
    raise SkipTest("dmp is not installed")
from bolero.datasets import make_minimum_jerk
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


def test_dmp_dimensions_do_not_match():
    beh = DMPBehavior()
    assert_raises_regexp(ValueError, "Input and output dimensions must match",
                         beh.init, 1, 2)

def test_shape_trajectory_imitate():
    n_step_evaluations = range(2,100)
    for n_steps in n_step_evaluations:
        n_task_dims = 1
        dt = 1.0/60  # 60 Hertz
        execution_time = dt * (n_steps - 1)  # -1 for shape(n_task_dims, n_steps)
        x0, g = np.zeros(1), np.ones(1)

        beh = DMPBehavior(execution_time, dt, 20)
        beh.init(3, 3)
        beh.set_meta_parameters(["x0", "g"], [x0, g])

        X_demo = np.empty((1, n_steps, 1))
        X_demo[0, :, 0] = np.linspace(0, 1, n_steps)
        assert_equal(n_steps, X_demo.shape[1])

        beh.imitate(X_demo, alpha=0.01)
        X, Xd, Xdd = beh.trajectory()

        assert_equal(X_demo[0, :].shape, X.shape)


def test_dmp_default_dmp():
    beh = DMPBehavior()
    beh.init(3 * n_task_dims, 3 * n_task_dims)

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


def test_dmp_get_set_params():
    beh = DMPBehavior()
    beh.init(3 * n_task_dims, 3 * n_task_dims)

    assert_equal(beh.get_n_params(), 50 * n_task_dims)
    params = beh.get_params()
    assert_array_equal(params, np.zeros(50 * n_task_dims))

    random_state = np.random.RandomState(0)
    expected_params = random_state.randn(50 * n_task_dims)
    beh.set_params(expected_params)

    actual_params = beh.get_params()
    assert_array_equal(actual_params, expected_params)


def test_dmp_from_config():
    beh = DMPBehavior(configuration_file=DMP_CONFIG_FILE)
    beh.init(18, 18)

    xva = np.zeros(18)
    beh.reset()
    t = 0
    while beh.can_step():
        eval_loop(beh, xva)
        t += 1

    assert_equal(t, 447)


def test_dmp_constructor_args():
    beh = DMPBehavior(execution_time=2)
    beh.init(3 * n_task_dims, 3 * n_task_dims)

    xva = np.zeros(3 * n_task_dims)
    beh.reset()
    t = 0
    while beh.can_step():
        eval_loop(beh, xva)
        t += 1

    assert_equal(t, 201)


def test_dmp_metaparameter_not_permitted():
    beh = DMPBehavior()
    beh.init(3, 3)
    assert_raises_regexp(ValueError, "Meta parameter .* is not allowed",
                         beh.set_meta_parameters, ["unknown"], [None])


def test_dmp_change_goal():
    beh = DMPBehavior()
    beh.init(3 * n_task_dims, 3 * n_task_dims)

    beh.set_meta_parameters(["g"], [np.ones(n_task_dims)])

    xva = np.zeros(3 * n_task_dims)
    beh.reset()
    while beh.can_step():
        eval_loop(beh, xva)
    for _ in range(30):  # Convergence
        eval_loop(beh, xva)

    assert_array_almost_equal(xva[:n_task_dims], np.ones(n_task_dims),
                              decimal=5)
    assert_array_almost_equal(xva[n_task_dims:-n_task_dims],
                              np.zeros(n_task_dims), decimal=4)
    assert_array_almost_equal(xva[-n_task_dims:], np.zeros(n_task_dims),
                              decimal=3)


def test_dmp_change_goal_velocity():
    beh = DMPBehavior()
    beh.init(3 * n_task_dims, 3 * n_task_dims)

    beh.set_meta_parameters(["gd"], [np.ones(n_task_dims)])

    xva = np.zeros(3 * n_task_dims)
    beh.reset()
    while beh.can_step():
        eval_loop(beh, xva)

    assert_array_almost_equal(xva[:n_task_dims], np.zeros(n_task_dims),
                              decimal=2)
    assert_array_almost_equal(xva[n_task_dims:-n_task_dims],
                              np.ones(n_task_dims), decimal=2)
    assert_array_almost_equal(xva[-n_task_dims:], np.zeros(n_task_dims),
                              decimal=0)


def test_dmp_change_execution_time():
    beh = DMPBehavior()
    beh.init(3 * n_task_dims, 3 * n_task_dims)

    beh.set_meta_parameters(["x0"], [np.ones(n_task_dims)])
    X1 = beh.trajectory()[0]
    beh.set_meta_parameters(["execution_time"], [2.0])
    X2 = beh.trajectory()[0]
    assert_equal(X2.shape[0], 201)
    assert_array_almost_equal(X1, X2[::2], decimal=3)


def test_dmp_change_weights():
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


def test_dmp_set_meta_params_before_init():
    beh = DMPBehavior()

    x0 = np.ones(n_task_dims) * 0.43
    g = np.ones(n_task_dims) * -0.21
    gd = np.ones(n_task_dims) * 0.12
    execution_time = 1.5

    beh.set_meta_parameters(["x0", "g", "gd", "execution_time"],
                            [x0, g, gd, execution_time])
    beh.init(3 * n_task_dims, 3 * n_task_dims)

    xva = np.zeros(3 * n_task_dims)
    xva[:n_task_dims] = x0

    beh.reset()
    t = 0
    while beh.can_step():
        eval_loop(beh, xva)
        t += 1

    assert_array_almost_equal(xva[:n_task_dims], g, decimal=3)
    assert_array_almost_equal(xva[n_task_dims:-n_task_dims], gd, decimal=2)
    assert_array_almost_equal(xva[-n_task_dims:], np.zeros(n_task_dims),
                              decimal=1)
    assert_equal(t, 151)


def test_dmp_more_steps_than_allowed():
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


def test_dmp_imitate():
    x0, g, execution_time, dt = np.zeros(1), np.ones(1), 1.0, 0.001

    beh = DMPBehavior(execution_time, dt, 20)
    beh.init(3, 3)
    beh.set_meta_parameters(["x0", "g"], [x0, g])

    X_demo = make_minimum_jerk(x0, g, execution_time, dt)[0]

    # Without regularization
    beh.imitate(X_demo)
    X = beh.trajectory()[0]
    assert_array_almost_equal(X_demo.T[0], X, decimal=2)

    # With alpha > 0
    beh.imitate(X_demo, alpha=1.0)
    X = beh.trajectory()[0]
    assert_array_almost_equal(X_demo.T[0], X, decimal=3)

    # Self-imitation
    beh.imitate(X.T[:, :, np.newaxis])
    X2 = beh.trajectory()[0]
    assert_array_almost_equal(X2, X, decimal=3)


def test_dmp_imitate_2d():
    x0, g, execution_time, dt = np.zeros(2), np.ones(2), 1.0, 0.001

    beh = DMPBehavior(execution_time, dt, 20)
    beh.init(6, 6)
    beh.set_meta_parameters(["x0", "g"], [x0, g])

    X_demo = make_minimum_jerk(x0, g, execution_time, dt)[0]

    # Without regularization
    beh.imitate(X_demo)
    X = beh.trajectory()[0]
    assert_array_almost_equal(X_demo.T[0], X, decimal=2)

    # With alpha > 0
    beh.imitate(X_demo, alpha=1.0)
    X = beh.trajectory()[0]
    assert_array_almost_equal(X_demo.T[0], X, decimal=3)

    # Self-imitation
    beh.imitate(X.T[:, :, np.newaxis])
    X2 = beh.trajectory()[0]
    assert_array_almost_equal(X2, X, decimal=3)


def test_dmp_imitate_pseudoinverse():
    x0, g, execution_time, dt = np.zeros(2), np.ones(2), 1.0, 0.01

    beh = DMPBehavior(execution_time, dt, 200)
    beh.init(6, 6)
    beh.set_meta_parameters(["x0", "g"], [x0, g])

    X_demo = make_minimum_jerk(x0, g, execution_time, dt)[0]

    beh.imitate(X_demo)
    X = beh.trajectory()[0]
    assert_array_almost_equal(X_demo.T[0], X, decimal=2)


def test_dmp_save_and_load():
    beh_original = DMPBehavior(execution_time=0.853, dt=0.001, n_features=10)
    beh_original.init(3 * n_task_dims, 3 * n_task_dims)

    x0 = np.ones(n_task_dims) * 1.29
    g = np.ones(n_task_dims) * 2.13
    beh_original.set_meta_parameters(["x0", "g"], [x0, g])

    xva = np.zeros(3 * n_task_dims)
    xva[:n_task_dims] = x0

    beh_original.reset()
    t = 0
    while beh_original.can_step():
        eval_loop(beh_original, xva)
        if t == 0:
            assert_array_almost_equal(xva[:n_task_dims], x0)
        t += 1
    assert_array_almost_equal(xva[:n_task_dims], g, decimal=3)
    assert_equal(t, 854)
    assert_equal(beh_original.get_n_params(), n_task_dims * 10)

    try:
        beh_original.save("tmp_dmp_model.yaml")
        beh_original.save_config("tmp_dmp_config.yaml")

        beh_loaded = DMPBehavior(configuration_file="tmp_dmp_model.yaml")
        beh_loaded.init(3 * n_task_dims, 3 * n_task_dims)
        beh_loaded.load_config("tmp_dmp_config.yaml")
    finally:
        if os.path.exists("tmp_dmp_model.yaml"):
            os.remove("tmp_dmp_model.yaml")
        if os.path.exists("tmp_dmp_config.yaml"):
            os.remove("tmp_dmp_config.yaml")

    xva = np.zeros(3 * n_task_dims)
    xva[:n_task_dims] = x0

    beh_loaded.reset()
    t = 0
    while beh_loaded.can_step():
        eval_loop(beh_loaded, xva)
        if t == 0:
            assert_array_almost_equal(xva[:n_task_dims], x0)
        t += 1
    assert_array_almost_equal(xva[:n_task_dims], g, decimal=3)
    assert_equal(t, 854)
    assert_equal(beh_loaded.get_n_params(), n_task_dims * 10)
