import numpy as np
from bolero.representation import DMPSequence
from bolero.controller import Controller
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_equal, assert_almost_equal


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
    return dmp_seq


def test_smoke():
    dmp_seq = create_dmp_seq(n_task_dims=1)

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
    assert_equal(dmp_seq.get_n_params(), 20)
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dmp_seq = create_dmp_seq(2)

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
