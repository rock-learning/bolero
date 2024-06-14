# Author: Hans Hohenfeld <hans.hohenfeld@dfki.de>

""" MOdel based Relative Entropy stochastic search """

from collections import deque

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from .optimizer import Optimizer

from ..utils.log import get_logger
from ..utils.validation import check_feedback, check_random_state

class QuadraticSurrogateModel(object):
    """ A Quadratic surrogate model for the objective function.

    The quadratic surrogate model is a local quadratic approximation of the value function
    learned from previous data. The data is fitted by a Bayesian linear regression model.
    """

    def __init__(self):
        """ The constructor """
        pass

    def fit(self, X, y):
        """ Fit Bayesian Linear Regression model.

        Parameters
        ----------
        X: array, shape (n_dims, n_samples)
	    Input data
        y: array, shape (n_samples,)
	    Target values

        """
        pass


class MOREOptimizer(Optimizer):
    """ The MORE Optimizer implements the MOdel based Relative Entropy stochastic search.

    The MORE stochastic search algorithm optimizes a search distribution with regards to
    the objective values while upper-bounding the KL-divergence and lower-bounding the
    entropy of the distribution:

    By using a quadratic surrogate model for the objective function, the algorithm can
    compute the integrals of the dual optimization problem analytically, satisfying the
    search distribution's bounds exactly.

    The surrogate model uses a Bayesian Linear Regression model to learn a local quadratic
    surrogate model for the objective function.

    References
    ----------
    .. [1] ABDOLMALEKI, Abbas, et al. Model-based relative entropy stochastic search. In: Advances
        in Neural Information Processing Systems. 2015. S. 3537-3545.
    """
    def __init__(self, initial_params=None, bounds=None,
                 epsilon=3.0, beta=-5000.0, train_freq=25,
                 n_samples_per_update=100,
                 log_to_stdout=False, log_to_file=False, random_state=None):
        self.initial_params = initial_params
        self.bounds = bounds

        self.epsilon = epsilon
        self.beta = beta

        self.log_to_stdout = log_to_stdout
        self.log_to_file = log_to_file

        self.random_state = random_state

        self.iteration = 0

        self.n_samples_per_update = n_samples_per_update
        self.train_freq = train_freq

        self.logger = None
        self.n_params = None

        self.params = None
        self.reward = None

        self.max_reward = -np.inf
        self.best_params = None

        self.reward_model = QuadraticSurrogateModel()

        self.hist_params = deque(maxlen=self.n_samples_per_update)
        self.hist_rewards = deque(maxlen=self.n_samples_per_update)

    def init(self, n_params):
        """Initialize the behavior search.

        Parameters
        ----------
        n_params : int
            dimension of the parameter vector
        """

        self.logger = get_logger(self, self.log_to_file, self.log_to_stdout)

        self.random_state = check_random_state(self.random_state)

        self.n_params = n_params

        if not self.initial_params:
            self.initial_params = np.zeros(self.n_params)
        else:
            self.initial_params = np.asarray(self.initial_params, dtype=np.float64)

            if len(self.initial_params) != self.n_params:
                raise ValueError("Initial parameter vector has wrong dimension. Is %d, expected %d!"
                                 % (len(self.initial_params), self.n_params))

        self.best_params = self.initial_params.copy()

    def get_next_parameters(self, params):
        """Get next individual/parameter vector for evaluation.

        Parameters
        ----------
        params : array_like, shape (n_params,)
            Parameter vector, will be modified
        """
        self._sample_from_policy()
        params[:] = self.params

    def set_evaluation_feedback(self, rewards):
        """Set feedbacks for the parameter vector.

        Parameters
        ----------
        rewards : list of float
            feedbacks for each step or for the episode, depends on the problem
        """
        self.reward = check_feedback(rewards, compute_sum=True)

        self.hist_params.append(self.params)
        self.hist_rewards.append(self.reward)

        self.iteration += 1

        # TODO: Surrogate Model / Policy update
        if self.iteration % self.train_freq == 0:
            X = np.asarray(self.hist_params)
            y = np.asarray(self.hist_rewards)

            self.reward_model.fit(X=X, y=y)

            self._update_policy()

        if self.reward > self.max_reward:
            self.max_reward = self.reward
            self.best_params = self.params

    def is_behavior_learning_done(self):
        """Check if the optimization is finished.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """
        return False

    def get_best_parameters(self):
        """Get best individual/parameter vector so far.

        Returns
        -------
        p : array_like, shape (n_params,)
            Best parameter vector so far
        """
        return self.best_params

    def _sample_from_policy(self):
        """ Sample new parameter vector from current search distribution. """
        pass

    def _update_policy(self):
        """ Update the search distribution """
        pass

    def _solve_dual_problem(self):
        """ Solve the dual optimization problem """


        def g(params):
            """ The dual function (eq. 4 in [1]) """
            eta = params[0]
            omega = params[1]

            constant_part = eta * self.epsilon - omega * self.beta

            result = constant_part + 0.5 # * .....

            return result


        x0 = [.0, .0]

        # TODO: Check for further constraints, especiall keeping eta large enoght
        # so that F stays positive definite!
        res = minimize(fun=g, x0=x0, method='L-BFGS-B', bounds=((0, None), (0, None)))
