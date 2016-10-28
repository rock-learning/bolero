# Author: Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import numpy as np
from .behavior import BlackBoxBehavior
import dmp


class DMPSequence(BlackBoxBehavior):
    """Sequence of DMPs.

    Each DMP is initialized at the last phase of its predecessor to ensure
    smooth transitions.

    Parameters
    ----------
    n_dmps : int, optional (default: 1)
        Number of DMPs in this sequence

    execution_times : array-like, shape (n_dmps,), optional (default: ones)
        Execution times of the DMPs

    n_features : array-like, shape (n_dmps,), optional (default: 50s)
        Number of RBF features for each dimension of each DMP.

    dt : float, optional (default: 0.01)
        Time between successive steps in seconds.

    subgoals : array-like, shape (n_dmps + 1, n_task_dims)
        Subgoals of the DMPs including the final goal and the
        initial state.

    learn_goal_velocities : bool, optional (default: True)
        Defines whether the parameter vector will include the velocities
        at the goals of the DMPs.

    initial_weights : list
        List of initial weight vectors for the DMPs
    """
    def __init__(self, n_dmps=1, execution_times=None, dt=0.01, n_features=None,
                 subgoals=None, learn_goal_velocities=False,
                 initial_weights=None):
        super(DMPSequence, self).__init__()
        self.n_dmps = n_dmps
        self.execution_times = execution_times
        self.dt = dt
        self.n_features = n_features
        self.subgoals = subgoals
        self.learn_goal_velocities = learn_goal_velocities
        self.initial_weights = initial_weights

    def init(self, n_inputs, n_outputs):
        """Initialize the behavior.

        Parameters
        ----------
        n_inputs : int
            number of inputs

        n_outputs : int
            number of outputs
        """
        if n_inputs != n_outputs:
            raise ValueError("Input and output dimensions must match, got "
                             "%d inputs and %d outputs" % (n_inputs, n_outputs))

        self.n_task_dims = n_inputs / 3

        if self.execution_times is None:
            self.execution_times = np.ones(self.n_dmps)
        self.execution_times = np.asarray(self.execution_times)
        if self.n_features is None:
            self.n_features = 50 * np.ones(self.n_dmps, dtype=int)
        if self.subgoals is None:
            self.subgoals = [np.zeros(self.n_task_dims)
                             for _ in range(self.n_dmps + 1)]
        else:
            self.subgoals = [np.asarray(g) for g in self.subgoals]

        self.n_steps_per_dmp = (self.execution_times / self.dt).astype(int) + 1
        self.n_weights_per_dmp = self.n_task_dims * self.n_features

        self.subgoal_velocities = [np.zeros(self.n_task_dims)
                                   for _ in range(self.n_dmps + 1)]

        self.n_weights = np.sum(self.n_weights_per_dmp)

        self.split_steps = np.cumsum(self.n_steps_per_dmp - 1)
        self.split_weights = np.cumsum(self.n_weights_per_dmp)

        self.alpha_z = []
        self.widths = []
        self.centers = []
        for i in range(self.n_dmps):
            alpha_z = dmp.calculate_alpha(0.01, self.execution_times[i], 0.0)
            widths = np.empty(self.n_features[i])
            centers = np.empty(self.n_features[i])
            dmp.initialize_rbf(widths, centers, self.execution_times[i],
                               0.0, 0.8, alpha_z)
            self.alpha_z.append(alpha_z)
            self.widths.append(widths)
            self.centers.append(centers)
        self.alpha_y = 25.0
        self.beta_y = self.alpha_y / 4.0
        if self.initial_weights is None:
            self.weights = [np.zeros(self.n_weights_per_dmp[i]
                            * self.n_task_dims)
                            for i in range(self.n_dmps)]
        else:
            self.weights = [w.ravel() for w in self.initial_weights]

        self.x0 = None
        self.g = None

        self.reset()

    def set_meta_parameters(self, keys, meta_parameters):
        """Set DMP meta parameters.

        Required meta-parameters

        x0 : array
            initial position
        g : array
            goal

        Parameters
        ----------
        keys : list of string
            names of meta-parameters
        meta_parameters : list of float
            values of meta-parameters
        """
        for key, meta_parameter in zip(keys, meta_parameters):
            setattr(self, key, meta_parameter)

        if self.x0 is not None:
            self.subgoals[0] = self.x0
        if self.g is not None:
            self.subgoals[-1] = self.g

    def set_inputs(self, inputs):
        """Set current system state.

        Parameters
        ----------
        inputs : array-like, shape (n_task_dims * 3)
            Position, velocitie and rotation of each dimension
        """
        assert len(inputs) == self.n_task_dims * 3
        self.last_y = inputs[:self.n_task_dims]
        self.last_yd = inputs[self.n_task_dims:-self.n_task_dims]
        self.last_ydd = inputs[-self.n_task_dims:]

    def get_outputs(self, outputs):
        """Get desired next system state.

        Parameters
        ----------
        outputs : array-like, shape (n_task_dims * 3)
            Position, velocitie and rotation of each dimension
        """
        assert len(outputs) == self.n_task_dims * 3
        outputs[:self.n_task_dims] = self.y
        outputs[self.n_task_dims:-self.n_task_dims] = self.yd
        outputs[-self.n_task_dims:] = self.ydd

    def step(self):
        """Compute next step."""
        if self.n_task_dims == 0:
            return

        dmp_idx = np.where(self.steps <= self.split_steps)[0][0]

        dmp.dmp_step(
            self.last_t, self.t,
            self.last_y, self.last_yd, self.last_ydd,
            self.y, self.yd, self.ydd,
            self.subgoals[dmp_idx + 1],
            self.subgoal_velocities[dmp_idx + 1],
            np.zeros(self.n_task_dims),
            self.subgoals[dmp_idx],
            self.subgoal_velocities[dmp_idx],
            np.zeros(self.n_task_dims),
            np.sum(self.execution_times[:dmp_idx + 1]),
            np.sum(self.execution_times[:dmp_idx]),
            self.weights[dmp_idx],
            self.widths[dmp_idx],
            self.centers[dmp_idx],
            self.alpha_y, self.beta_y, self.alpha_z[dmp_idx],
            0.001
        )

        if self.t == self.last_t:
            self.last_t = -1.0
        else:
            self.last_t = self.t
            self.t += self.dt

        self.steps += 1

    def get_n_params(self):
        """Get number of parameters."""
        n_params = self.n_weights
        n_params += (self.n_dmps - 1) * self.n_task_dims
        if self.learn_goal_velocities:
            n_params += self.n_dmps * self.n_task_dims
        return n_params

    def get_params(self):
        """Utility function: get currently optimizable parameters."""
        weights = np.concatenate([w for w in self.weights]).ravel()
        goals = np.concatenate(self.subgoals[1:-1])
        if self.learn_goal_velocities:
            goal_velocities = np.concatenate(self.subgoal_velocities)
            return np.hstack((weights, goals, goal_velocities))
        else:
            return np.hstack((weights, goals))

    def set_params(self, params):
        """Utility function: set currently optimizable parameters."""
        weights, goals, goal_vels = np.split(params, (self.n_weights,
            self.n_weights + (self.n_dmps - 1) * self.n_task_dims))
        G = np.split(goals, [i * self.n_task_dims
                             for i in range(1, self.n_dmps - 1)])
        self.weights = [w.ravel() for w in np.split(weights, self.split_weights)]

        for i in range(self.n_dmps - 1):
            self.subgoals[i + 1] = G[i]
        if self.learn_goal_velocities:
            self.subgoal_velocities = np.split(
                goal_vels, [i * self.n_task_dims
                            for i in xrange(1, self.n_dmps)])

    def set_subgoal(self, idx, subgoal):
        """Set subgoal manually.

        Parameters
        ----------
        idx: int
            index of the subgoal: 0 is the start of the DMP sequence,
            -1 would be the goal of the sequence

        subgoal: array-like, shape (n_task_dims,)
            subgoal at index 'idx'
        """
        assert np.abs(idx) < len(self.subgoals)
        assert self.n_task_dims == len(subgoal)
        self.subgoals[idx] = subgoal

    def get_subgoal(self, idx):
        """Get subgoal.

        Parameters
        ----------
        idx: int
            index of the subgoal: 0 is the start of the DMP sequence,
            -1 would be the goal of the sequence

        Returns
        -------
        subgoal: array-like, shape (n_task_dims,)
            subgoal at index 'idx'
        """
        assert np.abs(idx) < len(self.subgoals)
        return self.subgoals[idx]

    def set_subgoal_velocity(self, idx, subgoal_vel):
        """Set subgoal velocity manually.

        Parameters
        ----------
        idx: int
            index of the subgoal: 0 is the start of the DMP sequence,
            -1 would be the goal of the sequence

        subgoal_vel: array-like, shape (n_task_dims,)
            velocity at subgoal with index 'idx'
        """
        assert np.abs(idx) < len(self.subgoal_velocities)
        assert self.n_task_dims == len(subgoal_vel)
        self.subgoal_velocities[idx] = subgoal_vel

    def get_subgoal_velocity(self, idx):
        """Get subgoal.

        Parameters
        ----------
        idx: int
            index of the subgoal: 0 is the start of the DMP sequence,
            -1 would be the goal of the sequence

        Returns
        -------
        subgoal: array-like, shape (n_task_dims,)
            velocity at subgoal with index 'idx'
        """
        assert np.abs(idx) < len(self.subgoal_velocities)
        return self.subgoal_velocities[idx]


    def reset(self):
        """Reset DMP."""
        if self.x0 is None:
            self.last_y = np.zeros(self.n_task_dims)
        else:
            self.last_y = np.copy(self.subgoals[0])
        self.last_yd = np.copy(self.subgoal_velocities[0])
        self.last_ydd = np.zeros(self.n_task_dims)

        self.y = np.empty(self.n_task_dims)
        self.yd = np.empty(self.n_task_dims)
        self.ydd = np.empty(self.n_task_dims)

        self.last_t = 0.0
        self.t = 0.0
        self.steps = 0

    def can_step(self):
        """Returns true if step() can be called again, false otherwise."""
        return len(np.where(self.t <= self.split_steps)[0]) > 0

