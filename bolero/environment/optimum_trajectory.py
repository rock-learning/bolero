# Authors: Alexander Fabisch <afabisch@informatik.uni-bremen.de>
#          Marc Otto <maotto@uni-bremen.de>

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
        Radius of the obstacles that should be avoided (penalty is zero outside
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
        self.X *= np.nan
        self.Xd *= np.nan
        self.Xdd *= np.nan

    def get_num_inputs(self):
        """Get number of environment inputs.

        Returns
        ----------
        n : int
            number of environment inputs
        """
        return 3 * self.n_task_dims

    def get_num_outputs(self):
        """Get number of environment outputs.

        Returns
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

    def get_start_dist(self):
        """Get distance of trajectory start and desired start location.

        Returns
        -------
        start_dist : float
            start distance
        """
        start_dist = np.linalg.norm(self.x0 - self.X[0])
        self.logger.info("Distance to start: %.3f (* %.2f)"
                         % (start_dist, self.penalty_start_dist))
        return start_dist

    def get_goal_dist(self):
        """Get distance of trajectory end and desired goal location.

        Returns
        -------
        goal_dist : float
            goal distance
        """
        goal_dist = np.linalg.norm(self.g - self.X[-1])
        self.logger.info("Distance to goal: %.3f (* %.2f)"
                         % (goal_dist, self.penalty_goal_dist))
        self.logger.info("Goal: %s, last position: %s" % (self.g, self.X[-1]))
        return goal_dist

    def get_speed(self):
        """Get speed values during the performed movement.

        Returns
        -------
        speed : array-like, shape (n_steps,)
            the speed (scalar) at all previous timestamps
        """
        speed = np.sqrt(np.sum(self.Xd ** 2, axis=1))
        self.logger.info("Speed: %r" % speed)
        return speed

    def get_acceleration(self):
        """Get acceleration values during the performed movement.

        Returns
        -------
        acceleration : array-like, shape (n_steps,)
            the total acceleration (scalar) at all previous timestamps
        """
        acceleration = np.sqrt(np.sum(self.Xdd ** 2, axis=1))
        self.logger.info("Accelerations: %r" % acceleration)
        return acceleration

    def get_collision(self, obstacle_filter=None):
        """Get list of collisions with obstacles during the performed movement.

        Parameters
        ----------
        obstacle_filter : array-like, shape (n_desired_obstacles, 1)
            specify which obstacles cause collisions, e.g. set (0, 2) to
            exclude the second of three obstacles
        Returns
        -------
        collisions : array-like, shape (n_steps,)
            vector of values in range [0, 1] where distances above
            self.obstacle_dist result in 0 (no collision), and distance below
            are scaled linearly, so that 1 corresponds to an intersection.
        """
        if self.obstacles is None:
            return np.zeros(self.t)
        if obstacle_filter is None:
            obstacles = self.obstacles
        else:
            obstacles = np.asarray(self.obstacles)[obstacle_filter, :]
        distances = cdist(self.X, obstacles)
        self.logger.info("Distances to obstacles: %r" % distances)
        collision_penalties = np.maximum(0., 1.0 - distances /
                                         self.obstacle_dist)
        collisions = collision_penalties.sum(axis=1)
        return collisions

    def get_num_obstacles(self):
        """Get number of obstacles in environment.

        Returns
        ----------
        n : int
            number of obstacles
        """
        if self.obstacles is None:
            return 0
        return self.obstacles.shape[0]

    def get_feedback(self):
        """Get reward per timestamp based on weighted criteria (penalties)

        Returns
        -------
        rewards : array-like, shape (n_steps,)
            reward for every timestamp; non-positive values
        """
        rewards = np.zeros(self.t)
        if self.penalty_start_dist > 0.0:
            rewards[0] -= self.get_start_dist() * self.penalty_start_dist
        if self.penalty_goal_dist > 0.0:
            rewards[-1] -= self.get_goal_dist() * self.penalty_goal_dist
        if self.penalty_vel > 0.0:
            rewards -= self.get_speed() * self.penalty_vel
        if self.penalty_acc > 0.0:
            rewards -= self.get_acceleration() * self.penalty_acc
        if self.obstacles is not None and self.penalty_obstacle > 0.0:
            rewards -= self.penalty_obstacle * self.get_collision()
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
                ax.add_patch(Circle(np.copy(obstacle), self.obstacle_dist,
                                    ec="none", color="r"))


class OptimumTrajectoryCurbingObstacles(OptimumTrajectory):
    """Search trajectories with several criteria and curbing within obstacles.

    The parameter curbing_obstacles controls how much the next movement is
    slowed down, when the current position is within an obstacle. When several
    obstacles overlap, damping effects increase with the number of obstacles
    being hit. When the product of the number of obstacles being hit and the
    value of curbing_obstacles reaches one, no further movement is possible.

    Example: by default (curbing_obstacles = 0.5), hitting a single obstacles
    reduces following movements by 50% (1 * 0.5) until the obstacle is left.

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
        Radius of the obstacles that should be avoided (penalty is zero outside
        of this area)

    curbing_obstacles : float, optional (default: 0.5)
        Slow down the move when passing trough an obstacle. Passing through
        multiple obstacles at the same time increases the effect. Value should
        be in c = [0, 1]. Hitting k obstacles at the same time, leads to an
        inhibition of movement, if k * c >= 1, a slower movement if
        0 < k * c < 1 and otherwise (k * c = 0) doesn't reduce the movement.

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
                 curbing_obstacles=0.5, penalty_start_dist=0.0,
                 penalty_goal_dist=0.0, penalty_vel=0.0, penalty_acc=0.0,
                 penalty_obstacle=0.0, log_to_file=False, log_to_stdout=False):
        super(OptimumTrajectoryCurbingObstacles, self).__init__(
            x0=x0, g=g, execution_time=execution_time, dt=dt,
            obstacles=obstacles, obstacle_dist=obstacle_dist,
            penalty_start_dist=penalty_start_dist,
            penalty_goal_dist=penalty_goal_dist, penalty_vel=penalty_vel,
            penalty_acc=penalty_acc, penalty_obstacle=penalty_obstacle,
            log_to_file=log_to_file, log_to_stdout=log_to_stdout)
        self.curbing_obstacles = curbing_obstacles

    def init(self):
        """Initialize environment."""
        super(OptimumTrajectoryCurbingObstacles, self).init()
        self.damping = 0
        self.X *= np.nan
        self.Xd *= np.nan
        self.Xdd *= np.nan

    def reset(self):
        """Reset state of the environment."""
        super(OptimumTrajectoryCurbingObstacles, self).reset()
        self.damping = 0

    def set_inputs(self, values):
        """Set environment inputs, e.g. next action.

        Parameters
        ----------
        values : array,
            Inputs for the environment: positions, velocities and accelerations
            in that order, e.g. for n_task_dims=2 the order would be xxvvaa
        """
        if self.damping:
            if self.t == 0:
                raise RuntimeError("Damping can only occur after step action "
                                   "(self.t > 0) but self.t == 0")
            new_value_weight = max(0, (1 - self.damping))
            total_weight = self.damping + new_value_weight
            self.X[self.t, :] = (self.damping * self.X[self.t - 1, :] +
                                 new_value_weight * values[:self.n_task_dims])\
                / total_weight
            self.Xd[self.t, :] = (self.damping * self.Xd[self.t - 1, :] +
                                  new_value_weight *
                                  values[self.n_task_dims:-self.n_task_dims])\
                / total_weight
            self.Xdd[self.t, :] = (self.damping * self.Xdd[self.t - 1, :] +
                                   new_value_weight *
                                   values[-self.n_task_dims:])\
                / total_weight
        else:
            self.X[self.t, :] = values[:self.n_task_dims]
            self.Xd[self.t, :] = values[self.n_task_dims:-self.n_task_dims]
            self.Xdd[self.t, :] = values[-self.n_task_dims:]

    def step_action(self):
        """Execute step perfectly (unless obstacles are curbing)."""
        if self.curbing_obstacles:
            self.damping = (cdist(self.X[self.t:self.t+1, :], self.obstacles) <
                            self.obstacle_dist).ravel().sum()
            self.damping *= self.curbing_obstacles
        super(OptimumTrajectoryCurbingObstacles, self).step_action()