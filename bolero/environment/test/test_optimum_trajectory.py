import numpy as np
from bolero.environment import OptimumTrajectory
from nose.tools import assert_equal, assert_false, assert_true
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_dimensions():
    env = OptimumTrajectory(x0=np.zeros(3), g=np.ones(3))
    env.init()
    n_outputs = env.get_num_outputs()
    n_inputs = env.get_num_inputs()
    assert_equal(n_outputs, n_inputs)


def test_penalize_start():
    env = OptimumTrajectory(x0=np.zeros(1), g=np.ones(1), dt=1.0,
                            penalty_start_dist=4.0)
    env.init()
    n_outputs = env.get_num_outputs()
    n_inputs = env.get_num_inputs()

    xva = np.empty(n_outputs)
    env.reset()

    assert_false(env.is_evaluation_done())
    env.get_outputs(xva)
    xva[:] = 1.0
    env.set_inputs(xva)
    env.step_action()

    assert_false(env.is_evaluation_done())
    env.get_outputs(xva)
    assert_array_equal(xva, np.ones(3))
    env.set_inputs(xva)
    env.step_action()

    assert_true(env.is_evaluation_done())
    rewards = env.get_feedback()
    assert_array_almost_equal(rewards, np.array([-4.0, 0.0]))


def test_penalize_goal():
    env = OptimumTrajectory(x0=np.zeros(1), g=np.ones(1), dt=1.0,
                            penalty_goal_dist=5.0)
    env.init()
    n_outputs = env.get_num_outputs()
    n_inputs = env.get_num_inputs()

    xva = np.empty(n_outputs)
    env.reset()

    assert_false(env.is_evaluation_done())
    env.get_outputs(xva)
    assert_array_equal(xva, np.zeros(3))
    env.set_inputs(xva)
    env.step_action()

    assert_false(env.is_evaluation_done())
    env.get_outputs(xva)
    assert_array_equal(xva, np.zeros(3))
    env.set_inputs(xva)
    env.step_action()

    assert_true(env.is_evaluation_done())
    rewards = env.get_feedback()
    assert_array_almost_equal(rewards, np.array([0.0, -5.0]))


def test_penalize_velocity():
    env = OptimumTrajectory(x0=np.zeros(1), g=np.zeros(1), dt=1.0,
                            penalty_vel=6.0)
    env.init()
    n_outputs = env.get_num_outputs()
    n_inputs = env.get_num_inputs()

    xva = np.empty(n_outputs)
    env.reset()

    assert_false(env.is_evaluation_done())
    env.get_outputs(xva)
    xva = np.array([0.0, 1.0, 0.0])
    env.set_inputs(xva)
    env.step_action()

    assert_false(env.is_evaluation_done())
    env.get_outputs(xva)
    xva = np.array([0.0, 1.0, 0.0])
    env.set_inputs(xva)
    env.step_action()

    assert_true(env.is_evaluation_done())
    rewards = env.get_feedback()
    assert_array_almost_equal(rewards, -6.0 * np.ones(2))


def test_penalize_acceleration():
    env = OptimumTrajectory(x0=np.zeros(1), g=np.zeros(1), dt=1.0,
                            penalty_acc=7.0)
    env.init()
    n_outputs = env.get_num_outputs()
    n_inputs = env.get_num_inputs()

    xva = np.empty(n_outputs)
    env.reset()

    assert_false(env.is_evaluation_done())
    env.get_outputs(xva)
    xva = np.array([0.0, 0.0, 1.0])
    env.set_inputs(xva)
    env.step_action()

    assert_false(env.is_evaluation_done())
    env.get_outputs(xva)
    xva = np.array([0.0, 0.0, 1.0])
    env.set_inputs(xva)
    env.step_action()

    assert_true(env.is_evaluation_done())
    rewards = env.get_feedback()
    assert_array_almost_equal(rewards, -7.0 * np.ones(2))


def test_penalize_obstacles():
    obstacles = np.array([[0.0, 0.0]])
    env = OptimumTrajectory(dt=1.0, penalty_obstacle=8.0, obstacles=obstacles)
    env.init()
    n_outputs = env.get_num_outputs()
    n_inputs = env.get_num_inputs()

    xva = np.empty(n_outputs)
    env.reset()

    assert_false(env.is_evaluation_done())
    env.get_outputs(xva)
    env.set_inputs(xva)
    env.step_action()

    assert_false(env.is_evaluation_done())
    env.get_outputs(xva)
    env.set_inputs(xva)
    env.step_action()

    assert_true(env.is_evaluation_done())
    rewards = env.get_feedback()
    assert_array_almost_equal(rewards, -8.0 * np.ones(2))
