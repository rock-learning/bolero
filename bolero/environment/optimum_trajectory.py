# Authors: Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import numpy as np
from scipy.spatial.distance import cdist
from .environment import Environment
from bolero.utils.log import get_logger


class OptimumTrajectory(Environment):
    """Optimize a trajectory according to some criteria.

    Parameters
    ----------
    x0 : array-like, shape = (n_task_dims,), optional (default: [0, 0])
        Initial position.

    g : array-like, shape = (n_task_dims,), optional (default: [1, 1])
        Goal position.

    execution_time : float, optional (default: 1.0)
        Execution time in seconds

    dt : float, optional (default: 0.01)
        Time between successive steps in seconds.

    obstacles : array-like, shape (n_obstacles, n_task_dims) (default: None)
        List of obstacles.

    obstacle_dist : float, optional (default: 0.1)
        Distance that should be kept to the obstacles (penalty is zero outside
        of this area)

    penalty_start_dist : float, optional (default: 0)
        Penalty weight for distance to start at the beginning

    penalty_goal_dist : float, optional (default: 0)
        Penalty weight for distance to goal at the end

    penalty_vel : float, optional (default: 0)
        Penalty weight for velocities

    penalty_acc : float, optional (default: 0)
        Penalty weight for accelerations

    penalty_obstacle : float, optional (default: 0)
        Penalty weight for obstacle avoidance

    log_to_file: optional, boolean or string (default: False)
        Log results to given file, it will be located in the $BL_LOG_PATH

    log_to_stdout: optional, boolean (default: False)
        Log to standard output
    """
    def __init__(self, x0=np.zeros(2), g=np.ones(2), execution_time=1.0,
                 dt=0.01, obstacles=None, obstacle_dist=0.1,
                 penalty_start_dist=0.0, penalty_goal_dist=0.0,
                 penalty_vel=0.0, penalty_acc=0.0, penalty_obstacle=0.0,
                 log_to_file=False, log_to_stdout=False):
        self.x0 = x0
        self.g = g
        self.execution_time = execution_time
        self.dt = dt
        self.obstacles = obstacles
        self.obstacle_dist = obstacle_dist
        self.penalty_start_dist = penalty_start_dist
        self.penalty_goal_dist = penalty_goal_dist
        self.penalty_vel = penalty_vel
        self.penalty_acc = penalty_acc
        self.penalty_obstacle = penalty_obstacle
        self.log_to_file = log_to_file
        self.log_to_stdout = log_to_stdout

    def init(self):
        """Initialize environment."""
        self.x0 = np.asarray(self.x0)
        self.g = np.asarray(self.g)
        self.n_task_dims = self.x0.shape[0]
        self.logger = get_logger(self, self.log_to_file, self.log_to_stdout)

        n_steps = 1 + int(self.execution_time / self.dt)
        self.X = np.empty((n_steps, self.n_task_dims))
        self.Xd = np.empty((n_steps, self.n_task_dims))
        self.Xdd = np.empty((n_steps, self.n_task_dims))

    def reset(self):
        """Reset state of the environment."""
        self.t = 0

    def get_num_inputs(self):
        """Get number of environment inputs.

        Parameters
        ----------
        n : int
            number of environment inputs
        """
        return 3 * self.n_task_dims

    def get_num_outputs(self):
        """Get number of environment outputs.

        Parameters
        ----------
        n : int
            number of environment outputs
        """
        return 3 * self.n_task_dims

    def get_outputs(self, values):
        """Get environment outputs.

        Parameters
        ----------
        values : array
            Outputs of the environment: positions, velocities and accelerations
            in that order, e.g. for n_task_dims=2 the order would be xxvvaa
        """
        if self.t == 0:
            values[:self.n_task_dims] = np.copy(self.x0)
            values[self.n_task_dims:-self.n_task_dims] = np.zeros(
                self.n_task_dims)
            values[-self.n_task_dims:] = np.zeros(self.n_task_dims)
        else:
            values[:self.n_task_dims] = self.X[self.t - 1]
            values[self.n_task_dims:-self.n_task_dims] = self.Xd[self.t - 1]
            values[-self.n_task_dims:] = self.Xdd[self.t - 1]

    def set_inputs(self, values):
        """Set environment inputs, e.g. next action.

        Parameters
        ----------
        values : array,
            Inputs for the environment: positions, velocities and accelerations
            in that order, e.g. for n_task_dims=2 the order would be xxvvaa
        """
        self.X[self.t, :] = values[:self.n_task_dims]
        self.Xd[self.t, :] = values[self.n_task_dims:-self.n_task_dims]
        self.Xdd[self.t, :] = values[-self.n_task_dims:]

    def step_action(self):
        """Execute step perfectly."""
        self.t += 1

    def is_evaluation_done(self):
        """Check if the time is over.

        Returns
        -------
        finished : bool
            Is the episode finished?
        """
        return self.t * self.dt > self.execution_time

    def get_feedback(self):
        X = self.X
        Xd = self.Xd
        Xdd = self.Xdd

        rewards = np.zeros(self.t)

        if self.penalty_start_dist > 0.0:
            start_dist = np.linalg.norm(self.x0 - X[0])
            self.logger.info("Distance to start: %.3f (* %.2f)"
                             % (start_dist, self.penalty_start_dist))
            rewards[0] -= start_dist * self.penalty_start_dist

        if self.penalty_goal_dist > 0.0:
            goal_dist = np.linalg.norm(self.g - X[-1])
            self.logger.info("Distance to goal: %.3f (* %.2f)"
                             % (goal_dist, self.penalty_goal_dist))
            self.logger.info("Goal: %s, last position: %s" % (self.g, X[-1]))
            rewards[-1] -= goal_dist * self.penalty_goal_dist

        if self.penalty_vel > 0.0:
            velocities = np.sqrt(np.sum(Xd ** 2, axis=1))
            self.logger.info("Velocities: %r" % velocities)
            rewards -= velocities * self.penalty_vel

        if self.penalty_acc > 0.0:
            accelerations = np.sqrt(np.sum(Xdd ** 2, axis=1))
            self.logger.info("Accelerations: %r" % accelerations)
            rewards -= accelerations * self.penalty_acc

        if self.obstacles is not None and self.penalty_obstacle > 0.0:
            distances = cdist(X, self.obstacles)
            self.logger.info("Distances to obstacles: %r" % distances)
            collision_penalties = np.maximum(0, 1.0 - distances /
                                             self.obstacle_dist)
            rewards -= self.penalty_obstacle * collision_penalties.sum(axis=1)

        return rewards

    def is_behavior_learning_done(self):
        """Check if the behavior learning is finished.

        Returns
        -------
        finished : bool
            Always false
        """
        return False

    def get_maximum_feedback(self):
        """Returns the maximum sum of feedbacks obtainable."""
        return 0.0

    def plot(self, ax):
        """Plot a two-dimensional environment.

        Parameters
        ----------
        ax : Axis
            Matplotlib axis
        """
        if self.n_task_dims != 2:
            raise ValueError("Can only plot 2 dimensional environments")

        ax.scatter(self.x0[0], self.x0[1], c="r", s=100)
        ax.scatter(self.g[0], self.g[1], c="g", s=100)
        if self.obstacles is not None:
            from matplotlib.patches import Circle
            for obstacle in self.obstacles:
                ax.add_patch(Circle(obstacle, self.obstacle_dist, ec="none",
                                    color="r"))
