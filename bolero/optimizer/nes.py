# Author: Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import numpy as np
import scipy
from scipy.spatial.distance import pdist
from .optimizer import Optimizer
from ..utils.validation import check_random_state, check_feedback
from ..utils.log import get_logger


class XNESOptimizer(Optimizer):
    """Exponential Natural Evolution Strategies (xNES).

    See `Wikipedia <http://en.wikipedia.org/wiki/Natural_evolution_strategy>`_
    for details.

    Parameters
    ----------
    initial_params : array-like, shape = (n_params,), optional (default: 0s)
        Initial parameter vector.

    variance : float, optional (default: 1.0)
        Initial exploration variance.

    covariance : array-like, optional (default: None)
        Either a diagonal (with shape (n_params,)) or a full covariance matrix
        (with shape (n_params, n_params)). A full covariance can contain
        information about the correlation of variables.

    n_samples_per_update : integer, optional (default: 4+int(3*log(n_params)))
        Number of roll-outs that are required for a parameter update.

    bounds : array-like, shape (n_params, 2), optional (default: None)
        Upper and lower bounds for each parameter.

    maximize : boolean, optional (default: True)
        Maximize return or minimize cost?

    min_variance : float, optional (default: 2 * np.finfo(np.float).eps ** 2)
        Minimum variance before restart

    min_fitness_dist : float, optional (default: 2 * np.finfo(np.float).eps)
        Minimum distance between fitness values before restart

    max_condition : float optional (default: 1e7)
        Maximum condition of covariance matrix

    log_to_file: boolean or string, optional (default: False)
        Log results to given file, it will be located in the $BL_LOG_PATH

    log_to_stdout: boolean, optional (default: False)
        Log to standard output

    random_state : int or RandomState, optional (default: None)
        Seed for the random number generator or RandomState object.

    References
    ----------
    .. [1] Wierstra, D.; Schaul, T.; Glasmachers, T.; Sun, Y.; Peters, J.;
        Schmidhuber, J.
        Natural Evolution Strategies, Journal of Machine Learning Research,
        2014.
    """
    def __init__(
            self, initial_params=None, variance=1.0, covariance=None,
            n_samples_per_update=None, bounds=None, maximize=True,
            min_variance=2 * np.finfo(np.float).eps ** 2,
            min_fitness_dist=2 * np.finfo(np.float).eps, max_condition=1e7,
            log_to_file=False, log_to_stdout=False, random_state=None):
        self.initial_params = initial_params
        self.variance = variance
        self.covariance = covariance
        self.n_samples_per_update = n_samples_per_update
        self.bounds = bounds
        self.maximize = maximize
        self.min_variance = min_variance
        self.min_fitness_dist = min_fitness_dist
        self.max_condition = max_condition
        self.log_to_file = log_to_file
        self.log_to_stdout = log_to_stdout
        self.random_state = random_state

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
        self.it = 0

        if self.initial_params is None:
            self.initial_params = np.zeros(n_params)
        else:
            self.initial_params = np.asarray(self.initial_params).astype(
                np.float64, copy=True)
        if n_params != len(self.initial_params):
            raise ValueError("Number of dimensions (%d) does not match "
                             "number of initial parameters (%d)."
                             % (n_params, len(self.initial_params)))

        if self.covariance is None:
            self.covariance = np.eye(self.n_params)
        else:
            self.covariance = np.asarray(self.covariance).copy()
        if self.covariance.ndim == 1:
            self.covariance = np.diag(self.covariance)

        self.best_fitness = -np.inf
        self.best_fitness_it = self.it
        self.best_params = self.initial_params.copy()

        self._reinit()

    def _reinit(self):
        # Iteration of last reinitialization
        self.initial_it = self.it

        if self.n_samples_per_update is None:
            self.n_samples_per_update = 4 + int(3 * np.log(self.n_params))

        if self.bounds is not None:
            self.bounds = np.asarray(self.bounds)

        self.mean = self.initial_params.copy()

        self.noise = np.empty((self.n_samples_per_update, self.n_params))
        self.samples = np.empty((self.n_samples_per_update, self.n_params))
        self.fitness = np.empty(self.n_samples_per_update)

        self.A = np.linalg.cholesky(self.variance * self.covariance)
        self.learning_rate = (0.6 * (3.0 + np.log(self.n_params)) /
                              (self.n_params * np.sqrt(self.n_params)))

        utilities = np.maximum(np.log1p(self.n_samples_per_update / 2.0) -
                               np.log(self.n_samples_per_update -
                                      np.arange(self.n_samples_per_update)), 0)
        utilities /= np.sum(utilities)
        self.utilities = utilities - 1.0 / self.n_samples_per_update

        self._sample()

    def _sample(self):
        self.noise[:, :] = self.random_state.randn(
            self.n_samples_per_update, self.n_params)
        self.samples[:, :] = self.noise.dot(self.A.T) + self.mean
        if self.bounds is not None:
            np.clip(self.samples, self.bounds[:, 0], self.bounds[:, 1],
                    out=self.samples)

    def get_next_parameters(self, params):
        """Get next individual/parameter vector for evaluation.

        Parameters
        ----------
        params : array_like, shape (n_params,)
            Parameter vector, will be modified
        """
        k = self.it % self.n_samples_per_update
        params[:] = self.samples[k]

    def set_evaluation_feedback(self, feedback):
        """Set feedbacks for the parameter vector.

        Parameters
        ----------
        feedback : list of float
            feedbacks for each step or for the episode, depends on the problem
        """
        k = self.it % self.n_samples_per_update
        self.fitness[k] = check_feedback(feedback, compute_sum=True)
        if not self.maximize:
            self.fitness[k] *= -1

        if self.fitness[k] >= self.best_fitness:
            self.best_fitness = self.fitness[k]
            self.best_fitness_it = self.it
            self.best_params[:] = self.samples[k]

        self.it += 1

        if self.log_to_stdout or self.log_to_file:
            self.logger.info("[XNES] Iteration #%d, fitness: %g"
                             % (self.it, self.fitness[k]))

        if (self.it - self.initial_it) % self.n_samples_per_update == 0:
            self._update(self.samples, self.fitness, self.it)
            self._sample()

    def _update(self, samples, fitness, it):
        # Sample weights for mean recombination
        ranking = np.argsort(self.fitness, axis=0)  # Rank -> sample
        ranking = np.argsort(ranking, axis=0)       # Sample -> rank
        utilities = self.utilities[ranking]

        self.mean += self.A.dot(utilities.dot(self.noise))
        cov_gradient = np.sum([u * np.outer(s, s)
                               for s, u in zip(self.noise, utilities)], axis=0)
        # We don't need to subtract u * I because the utilities sum up to 0,
        # hence, the term cancels out
        self.A = np.dot(self.A, scipy.linalg.expm(0.5 * self.learning_rate *
                                                  cov_gradient))

    def is_behavior_learning_done(self):
        """Check if the optimization is finished.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """
        if self.it <= self.n_samples_per_update:
            return False

        if not np.all(np.isfinite(self.fitness)):
            return True

        # Check for invalid values
        if not (np.all(np.isfinite(self.A)) and
                np.all(np.isfinite(self.mean))):
            self.logger.info("Stopping: infs or nans" % self.var)
            return True

        if (self.min_variance is not None and
                np.max((self.A ** 2).sum(axis=1)) <= self.min_variance):
            self.logger.info("Stopping: %g < min_variance"
                             % np.max((self.A ** 2).sum(axis=1)))
            return True

        max_dist = np.max(pdist(self.fitness[:, np.newaxis]))
        if max_dist < self.min_fitness_dist:
            self.logger.info("Stopping: %g < min_fitness_dist" % max_dist)
            return True

        cov_diag = (self.A ** 2).sum(axis=1)
        if (self.max_condition is not None and
                np.max(cov_diag) > self.max_condition * np.min(cov_diag)):
            self.logger.info("Stopping: %g / %g > max_condition"
                             % (np.max(cov_diag), np.min(cov_diag)))
            return True

        return False

    def get_best_parameters(self, method="best"):
        """Get the best parameters.

        Parameters
        ----------
        method : string, optional (default: 'best')
            Either 'best' or 'mean'

        Returns
        -------
        best_params : array-like, shape (n_params,)
            Best parameters
        """
        if method == "best":
            return self.best_params
        else:
            return self.mean

    def get_best_fitness(self):
        """Get the best observed fitness.

        Returns
        -------
        best_fitness : float
            Best fitness (sum of feedbacks) so far. Corresponds to the
            parameters obtained by get_best_parameters(method='best'). For
            maximize=True, this is the highest observed fitness, and for
            maximize=False, this is the lowest observed fitness.
        """
        if self.maximize:
            return self.best_fitness
        else:
            return -self.best_fitness

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["logger"]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.logger = get_logger(self, self.log_to_file, self.log_to_stdout)

