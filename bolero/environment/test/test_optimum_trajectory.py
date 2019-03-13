import numpy as np
from bolero.environment import OptimumTrajectory, \
    OptimumTrajectoryCurbingObstacles
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


def _run_curbing_obstacles(obstacles, obstacle_dist, curbing_obs):
    # common evaluation method for curbing obstacles environment
    relative_movement_target = np.array([0.1, 0.1, 0, 0, 0, 0])
    env = OptimumTrajectoryCurbingObstacles(x0=np.zeros(2), g=np.ones(2),
                                            execution_time=1.0, dt=0.05,
                                            obstacles=obstacles,
                                            obstacle_dist=obstacle_dist,
                                            curbing_obstacles=curbing_obs)
    env.init()
    n_outputs = env.get_num_outputs()

    xva = np.empty(n_outputs)
    env.reset()

    assert_false(env.is_evaluation_done())
    env.get_outputs(xva)
    assert_array_equal(xva, np.zeros(6))

    for _ in range(10):
        # predict next state
        if np.linalg.norm(xva[:2] - obstacles[0]) < obstacle_dist:
            # in obstacle
            expected_move = (1. - curbing_obs) * relative_movement_target
        else:
            # free movement
            expected_move = relative_movement_target
        expected_new_position = xva[:2] + expected_move[:2]

        # set next target position
        env.set_inputs(xva + relative_movement_target)
        env.step_action()
        env.get_outputs(xva)

        # confirm prediction
        assert_array_almost_equal(xva[:2], expected_new_position,
                                  err_msg="x: actual position, y: expectation")
    return xva


def test_curbing_obstacles_full_stop():
    # tests that trajectory is stopped after obstacle contact
    obstacles = [[0.5, 0.5]]
    obstacle_dist = 0.1
    curbing_obstacles = 1.  # <- full stop at obstacle contact
    xva = _run_curbing_obstacles(obstacles, obstacle_dist, curbing_obstacles)
    
    # the final position needs to be within the obstacle
    assert_true(np.linalg.norm(xva[:2] - obstacles[0]) < obstacle_dist)


def test_curbing_obstacles_slowed_down():
    # tests that passing through an obstacle slows down according to settings
    obstacles = [[0.5, 0.5]]
    obstacle_dist = 0.1

    curbing_obstacles = 0.5  # <- half speed following obstacle contact
    xva_5 = _run_curbing_obstacles(obstacles, obstacle_dist, curbing_obstacles)

    curbing_obstacles = 0.2  # <- half speed following obstacle contact
    xva_2 = _run_curbing_obstacles(obstacles, obstacle_dist, curbing_obstacles)

    # with lower curbing factor, final position should be further from origin
    assert_true(np.linalg.norm(xva_2[:2]) > np.linalg.norm(xva_5[:2]))
