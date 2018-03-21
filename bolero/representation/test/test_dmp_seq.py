import numpy as np
try:
    from bolero.representation import DMPSequence
except ImportError:
    from nose import SkipTest
    raise SkipTest("dmp is not installed")
from bolero.controller import Controller
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_equal, assert_almost_equal, assert_raises_regexp


def create_dmp_seq(n_task_dims=1):
    execution_times = [0.2, 0.3, 0.5]
    n_features = [5, 5, 5]
    subgoals = [[0] * n_task_dims, [0.5] * n_task_dims,
                [1.0] * n_task_dims, [2] * n_task_dims]
    dmp_seq = DMPSequence(3, execution_times, 0.01, n_features, subgoals,
                          learn_goal_velocities=True)
    dmp_seq.init(3 * n_task_dims, 3 * n_task_dims)
    dmp_seq.set_meta_parameters(["x0", "g"], [np.zeros(n_task_dims),
                                              2 * np.ones(n_task_dims)])
    return dmp_seq, subgoals


def test_dimensions_mismatch():
    execution_times = [0.2, 0.3, 0.5]
    n_features = [5, 5, 5]
    subgoals = [[0], [0.5], [1.0], [2]]
    dmp_seq = DMPSequence(3, execution_times, 0.01, n_features, subgoals,
                          learn_goal_velocities=True)
    assert_raises_regexp(ValueError, "Input and output dimensions",
                         dmp_seq.init, 3, 4)


def test_make_execution_times():
    n_features = [5, 5, 5]
    subgoals = [[0], [0.5], [1.0], [2]]
    dmp_seq = DMPSequence(3, None, 0.01, n_features, subgoals,
                          learn_goal_velocities=True)
    dmp_seq.init(3, 3)
    assert_array_almost_equal(dmp_seq.execution_times, [1, 1, 1])


def test_make_n_features():
    execution_times = [0.2, 0.3, 0.5]
    subgoals = [[0], [0.5], [1.0], [2]]
    dmp_seq = DMPSequence(3, execution_times, 0.01, None, subgoals,
                          learn_goal_velocities=True)
    dmp_seq.init(3, 3)
    assert_array_almost_equal(dmp_seq.n_features, [50, 50, 50])


def test_make_subgoals():
    execution_times = [0.2, 0.3, 0.5]
    n_features = [5, 5, 5]
    dmp_seq = DMPSequence(3, execution_times, 0.01, n_features, None,
                          learn_goal_velocities=True)
    dmp_seq.init(3, 3)
    assert_equal(len(dmp_seq.subgoals), 4)


def test_initial_weights():
    execution_times = [0.2, 0.3, 0.5]
    n_features = [5, 5, 5]
    subgoals = [[0], [0.5], [1.0], [2]]
    initial_weights = [np.ones(5), np.ones(5), np.ones(5)]
    dmp_seq = DMPSequence(3, execution_times, 0.01, n_features, subgoals,
                          learn_goal_velocities=True,
                          initial_weights=initial_weights)
    dmp_seq.init(3, 3)
    assert_equal(len(dmp_seq.weights), 3)


def test_empty_metaparameters():
    execution_times = [0.2, 0.3, 0.5]
    n_features = [5, 5, 5]
    subgoals = [[0], [0.5], [1.0], [2]]
    dmp_seq = DMPSequence(3, execution_times, 0.01, n_features, subgoals,
                          learn_goal_velocities=True)
    dmp_seq.init(3, 3)
    dmp_seq.set_meta_parameters([], [])


def test_n_params():
    execution_times = [0.2, 0.3, 0.5]
    n_features = [5, 5, 5]
    subgoals = [[0], [0.5], [1.0], [2]]
    initial_weights = [np.ones(5), np.ones(5), np.ones(5)]
    dmp_seq = DMPSequence(3, execution_times, 0.01, n_features, subgoals,
                          learn_goal_velocities=True,
                          initial_weights=initial_weights)
    dmp_seq.init(3, 3)
    assert_equal(len(dmp_seq.get_params()), 3 * 5 + 2 * 1 + 4 * 1)
    assert_equal(dmp_seq.get_n_params(), 3 * 5 + 2 * 1 + 4 * 1)
    dmp_seq.set_params(dmp_seq.get_params())

    dmp_seq = DMPSequence(3, execution_times, 0.01, n_features, subgoals,
                          learn_goal_velocities=False,
                          initial_weights=initial_weights)
    dmp_seq.init(3, 3)
    assert_equal(len(dmp_seq.get_params()), 3 * 5 + 2)
    assert_equal(dmp_seq.get_n_params(), 3 * 5 + 2)
    dmp_seq.set_params(dmp_seq.get_params())


def test_no_dimensions():
    dmp_seq = DMPSequence()
    dmp_seq.init(0, 0)
    dmp_seq.set_meta_parameters([], [])
    io = np.empty(0)
    dmp_seq.set_inputs(io)
    dmp_seq.step()
    dmp_seq.get_outputs(io)


def test_smoke():
    dmp_seq, _ = create_dmp_seq(n_task_dims=1)

    controller = Controller({"Controller":
                             {"record_inputs": True},
                             "Environment":
                             {"type": "bolero.environment.OptimumTrajectory",
                              "x0": np.zeros(1),
                              "g": np.ones(1),
                              "execution_time": 1.0,
                              "dt": 0.01}})
    controller.episode_with(dmp_seq)

    params = np.random.randn(dmp_seq.get_n_params())
    dmp_seq.set_params(params)
    assert_equal(dmp_seq.get_n_params(), dmp_seq.get_params().size)
    assert_equal(dmp_seq.n_weights, 15)
    assert_equal(dmp_seq.get_n_params(), 21)
    params_copy = dmp_seq.get_params()
    assert_array_almost_equal(params, params_copy)
    new_subgoal = [0.5]
    dmp_seq.set_subgoal(2, new_subgoal)
    subgoal = dmp_seq.get_subgoal(2)
    assert_array_almost_equal(new_subgoal, subgoal)
    new_subgoal_velocity = [0.0]
    dmp_seq.set_subgoal_velocity(2, new_subgoal_velocity)
    subgoal_velocity = dmp_seq.get_subgoal_velocity(2)
    assert_array_almost_equal(new_subgoal_velocity, subgoal_velocity)

    X = np.array(controller.inputs_[0])
    assert_equal(len(X), 101)
    assert_almost_equal(X[0, 0], 0.0)
    assert_almost_equal(X[20, 0], 0.5, places=2)
    assert_almost_equal(X[50, 0], 1.0, places=2)
    assert_almost_equal(X[100, 0], 2.0, places=2)


def test_trajectory_generation():
    dmp_seq, _ = create_dmp_seq(n_task_dims=1)
    traj = dmp_seq.trajectory()[0]
    subgoal = dmp_seq.get_subgoal(0)
    assert_almost_equal(traj[0, 0], subgoal)
    subgoal = dmp_seq.get_subgoal(1)
    assert_almost_equal(traj[20, 0], subgoal, places=2)
    subgoal = dmp_seq.get_subgoal(2)
    assert_almost_equal(traj[50, 0], subgoal, places=2)
    subgoal = dmp_seq.get_subgoal(3)
    assert_almost_equal(traj[100, 0], subgoal, places=2)


def test_set_params_with_subgoal_velocities():
    test_task_dims = range(1, 100, 1)
    for n_task_dims in test_task_dims:
        dmp_seq, subgoals = create_dmp_seq(n_task_dims=n_task_dims)
        dmp_seq.set_params(dmp_seq.get_params())
        for i in range(len(subgoals)):
            assert_equal(len(dmp_seq.get_subgoal_velocity(i)), n_task_dims)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dmp_seq, _ = create_dmp_seq(2)

    controller = Controller({"Controller":
                             {"record_outputs": True},
                             "Environment":
                             {"type": "bolero.environment.OptimumTrajectory",
                              "x0": np.zeros(2),
                              "g": np.ones(2),
                              "execution_time": 1.0,
                              "dt": 0.01}},
                            record_inputs=True)
    controller.episode_with(dmp_seq)
    X = np.array(controller.outputs_)[0]

    plt.figure()
    plt.plot(X[:, 0], X[:, 1])
    plt.figure()
    plt.plot(X[:, 2])
    plt.figure()
    plt.plot(X[:, 3])
    plt.show()
