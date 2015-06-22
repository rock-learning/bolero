# Author: Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import numpy as np
from .optimizer import Optimizer
from ..utils.validation import check_random_state


class NoOptimizer(Optimizer):
    """No optimizer.

    Implements the optimizer interface but does not modify any parameters.

    Parameters
    ----------
    initial_params : array-like, shape = (n_params,), optional (default: [0, 0])
        Initial parameter vector.
    """
    def __init__(self, initial_params=None, **kwargs):
        self.initial_params = initial_params

    def init(self, dimension):
        if self.initial_params is None:
            self.initial_params = np.zeros(dimension)
        else:
            self.initial_params = np.asarray(self.initial_params).astype(
                np.float64, copy=True)
        if dimension != len(self.initial_params):
            raise ValueError("Number of dimensions (%d) does not match "
                             "number of initial parameters (%d)."
                             % (dimension, len(self.initial_params)))
        self.best_params = np.asarray(self.initial_params).copy()

    def get_next_parameters(self, p, explore=True):
        p[:] = self.best_params

    def set_evaluation_feedback(self, rewards):
        pass

    def get_best_parameters(self):
        return self.best_params

    def is_behavior_learning_done(self):
        return False


class RandomOptimizer(Optimizer):
    """Random optimizer.

    Parameters
    ----------
    initial_params : array-like, shape = (n_params,), optional (default: [0, 0])
        Initial parameter vector.

    covariance : array-like, shape = (n_params,), optional (default: I)
        Exploration covariance.

    random_state : int, optional
        Seed for the random number generator.
    """
    def __init__(self, initial_params=None, covariance=None, random_state=None,
                 **kwargs):
        self.initial_params = initial_params
        self.covariance = covariance
        self.random_state = random_state

    def init(self, dimension):
        if self.initial_params is None:
            self.initial_params = np.zeros(dimension)
        else:
            self.initial_params = np.asarray(self.initial_params).astype(
                np.float64, copy=True)
        if dimension != len(self.initial_params):
            raise ValueError("Number of dimensions (%d) does not match "
                             "number of initial parameters (%d)."
                             % (dimension, len(self.initial_params)))
        self.best_params = np.asarray(self.initial_params).copy()
        self.params = np.zeros(dimension)
        if self.covariance is None:
            self.covariance = np.eye(dimension)
        self.random_state = check_random_state(self.random_state)
        self.best_reward = -np.inf

    def get_next_parameters(self, p):
        self.params = self.random_state.multivariate_normal(
            self.initial_params, self.covariance, size=1)[0]
        p[:] = self.params

    def set_evaluation_feedback(self, rewards):
        r = np.sum(rewards)
        if r > self.best_reward:
            self.best_reward = r
            self.best_params = self.params.copy()

    def get_best_parameters(self):
        return self.best_params

    def is_behavior_learning_done(self):
        return False
