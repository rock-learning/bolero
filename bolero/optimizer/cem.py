import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from bolero.optimizer import Optimizer
from bolero.optimizer.cmaes import _bound
from bolero.utils.validation import check_random_state, check_feedback

class CEMOptimizer(Optimizer):
    """Cross entropy method.

    See 'Wikipedia <https://en.wikipedia.org/wiki/Cross-entropy_method>'_ for details.

    Parameters
    ----------
    initial_params : array-like, shape = (n_params,), optional (default: [0, 0])
        Initial parameter vector.

    covariance : array-like, shape = (n_params,), optional (default: I)
        Exploration covariance.

    n_samples_per_update : integer, optional (default: 4+int(3*log(n_params)))
        Number of roll-outs that are required for a parameter update.

    bounds : array-like, shape (n_params, 2), optional (default: None)
        Upper and lower bounds for each parameter.

    maximize : boolean, optional (default: True)
        Maximize return or minimize cost?

    elite_frac:float, optional (default: 0.5(50 %))
        Best candidate solutions for update

    min_variance : float, optional (default: 2 * np.finfo(np.float).eps ** 2)
        Minimum variance as a stopping criteria for behaviour learning

    min_fitness_dist : float, optional (default: 2 * np.finfo(np.float).eps)
        Minimum distance between fitness values

    random_state : int, optional
        Seed for the random number generator.
    """

    def __init__(self, initial_params=None, covariance=None,
                 n_samples_per_update=None, elite_frac=0.5,
                 bounds=None, min_fitness_dist=2 * np.finfo(np.float).eps,
                 min_variance=2 * np.finfo(np.float).eps ** 2,
                 random_state=None, maximize=True,
                 **kwargs):
        self.initial_params = initial_params
        self.covariance = covariance
        self.n_samples_per_update = n_samples_per_update
        self.maximize = maximize
        self.elite_frac = elite_frac
        self.bounds = bounds
        self.min_variance = min_variance
        self.min_fitness_dist = min_fitness_dist
        self.random_state = random_state
        

    def init(self, n_params):
        """Initialize the behavior search.

        Parameters
        ----------
        n_params : int
            dimension of the parameter vector
        """

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
        if self.elite_frac <= 0 or self.elite_frac >= 1:
            raise ValueError("Elite fraction should be positive and in range (0.0,1.0)")
        if self.covariance is None:
            self.covariance = np.eye(self.n_params)
        else:
            self.covariance = np.asarray(self.covariance).copy()
        if self.covariance.ndim == 1:
            self.covariance = np.diag(self.covariance)

        self.best_fitness = np.inf
        self.best_fitness_it = self.it
        self.best_params = self.initial_params.copy()

        self.initial_it = self.it

        if self.n_samples_per_update is None:
            self.n_samples_per_update = 4 + int(3 * np.log(self.n_params))
        if self.bounds is not None:
            self.bounds = np.asarray(self.bounds)
        self.mean = self.initial_params.copy()
        self.cov = self.covariance.copy()

        self.samples = self._sample(self.n_samples_per_update)
        self.fitness = np.empty(self.n_samples_per_update)

    def _sample(self, n_samples):
        samples = self.random_state.multivariate_normal(
            self.mean, self.cov, size=n_samples)
        _bound(self.bounds, samples)
        return samples

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
        if self.maximize:
            self.fitness[k] *= -1

        if self.fitness[k] <= self.best_fitness:
            self.best_fitness = self.fitness[k]
            self.best_fitness_it = self.it
            self.best_params[:] = self.samples[k]

        self.it += 1

        if (self.it - self.initial_it) % self.n_samples_per_update == 0:
            self._update(self.samples, self.fitness)

    def _update(self, samples, fitness):
        # -> Update sample distribution mean and cov
        self.last_mean = self.mean
        self.last_cov = self.cov
        elite_sol = int(self.n_samples_per_update * self.elite_frac)
        ranking = np.argsort(fitness, axis=0)
        update_samples = samples[ranking[:elite_sol]]
        self.mean = np.mean(update_samples, axis=0) 
        self.cov = np.cov(update_samples, rowvar=False)
        self.samples = self._sample(self.n_samples_per_update)

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
            return -self.best_fitness
        else:
            return self.best_fitness

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
        if not (np.all(np.isfinite(self.cov)) and
                np.all(np.isfinite(self.mean))):
            return True
        # check if variance <min_variance
        if (self.min_variance is not None and
                np.max(np.diag(self.cov)) <= self.min_variance):
            return True
        # check if distance between fitness values < min_fitness_dist
        max_dist = np.max(pdist(self.fitness[:, np.newaxis]))
        if max_dist < self.min_fitness_dist:
            return True
        return False
