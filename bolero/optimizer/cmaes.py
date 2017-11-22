# Author: Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import numpy as np
import warnings
from scipy.spatial.distance import pdist
from .optimizer import Optimizer
from ..utils.validation import check_random_state, check_feedback
from ..utils.log import get_logger


def _bound(bounds, samples):
    """Apply boundaries to samples.

    Parameters
    ----------
    bounds : array-like, shape (n_params, 2)
        Boundaries, bounds[:, 0] are the lower boundaries and
        bounds[:, 1] are the upper boundaries

    samples : array-like, shape (n_samples, n_params)
        Samples from the search distribution. Will be modified so that they
        are within the boundaries.
    """
    if bounds is not None:
        # TODO vectorize?
        for k in range(len(samples)):
            samples[k] = np.maximum(samples[k], bounds[:, 0])
            samples[k] = np.minimum(samples[k], bounds[:, 1])


def inv_sqrt(cov):
    """Compute inverse square root of a covariance matrix."""
    cov = np.triu(cov) + np.triu(cov, 1).T
    D, B = np.linalg.eigh(cov)
    # HACK: avoid numerical problems
    D = np.maximum(D, np.finfo(np.float).eps)
    D = np.sqrt(D)
    return B.dot(np.diag(1.0 / D)).dot(B.T), B, D


class CMAESOptimizer(Optimizer):
    """Covariance Matrix Adaptation Evolution Strategy.

    See `Wikipedia <http://en.wikipedia.org/wiki/CMA-ES>`_ for details.

    Plain CMA-ES is considered to be useful for

    * non-convex,
    * non-separable,
    * ill-conditioned,
    * or noisy

    objective functions. However, in some cases CMA-ES will be outperformed
    by other methods:

    * if the search space dimension is very small (e.g. less than 5),
      downhill simplex or surrogate-assisted methods will be better
    * easy functions (separable, nearly quadratic, etc.) will usually be
      solved faster by NEWUOA
    * multimodal objective functions require restart strategies

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

    active : bool, optional (default: False)
        Active CMA-ES (aCMA-ES) with negative weighted covariance matrix
        update

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
    """
    def __init__(
            self, initial_params=None, variance=1.0, covariance=None,
            n_samples_per_update=None, active=False, bounds=None, maximize=True,
            min_variance=2 * np.finfo(np.float).eps ** 2,
            min_fitness_dist=2 * np.finfo(np.float).eps, max_condition=1e7,
            log_to_file=False, log_to_stdout=False, random_state=None):
        self.initial_params = initial_params
        self.variance = variance
        self.covariance = covariance
        self.n_samples_per_update = n_samples_per_update
        self.active = active
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
        self.eigen_decomp_updated = 0

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

        self.best_fitness = np.inf
        self.best_fitness_it = self.it
        self.best_params = self.initial_params.copy()

        self._reinit()

    def _reinit(self):
        # Iteration of last reinitialization
        self.initial_it = self.it

        self.var = self.variance

        if self.n_samples_per_update is None:
            self.n_samples_per_update = 4 + int(3 * np.log(self.n_params))

        if self.bounds is not None:
            self.bounds = np.asarray(self.bounds)

        self.mean = self.initial_params.copy()
        self.cov = self.covariance.copy()

        self.samples = self._sample(self.n_samples_per_update)
        self.fitness = np.empty(self.n_samples_per_update)

        # Sample weights for mean recombination
        self.mu = self.n_samples_per_update / 2.0
        self.weights = (np.log(self.mu + 0.5) -
                        np.log1p(np.arange(int(self.mu))))
        self.mu = int(self.mu)
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)

        # Time constant for cumulation of the covariance
        self.cc = ((4 + self.mueff / self.n_params) /
                   (self.n_params + 4 + 2 * self.mueff / self.n_params))
        # Time constant for cumulation for sigma control
        self.cs = (self.mueff + 2) / (self.n_params + self.mueff + 5)
        # Learning rate for rank-one update
        self.c1 = 2 / ((self.n_params + 1.3) ** 2 + self.mueff)
        # Learning rate for rank-mu update
        self.cmu = (np.min((1 - self.c1, 2 * self.mueff - 2 +
                            1.0 / self.mueff)) /
                    ((self.n_params + 2) ** 2 + self.mueff))
        # Damping for sigma
        self.damps = 1 + 2 * np.max((0, np.sqrt((self.mueff - 1) /
                                    (self.n_params + 1)) - 1)) + self.cs

        # Misc constants
        self.ps_update_weight = np.sqrt(self.cs * (2 - self.cs) * self.mueff)
        self.hsig_threshold = 2 + 4.0 / (self.n_params + 1)
        self.eigen_update_freq = (self.n_samples_per_update /
                                  ((self.c1 + self.cmu) * self.n_params * 10))

        # Evolution path for covariance
        self.pc = np.zeros(self.n_params)
        # Evolution path for sigma
        self.ps = np.zeros(self.n_params)

        if self.active:
            self.alpha_old = 0.5
            self.neg_cmu = ((1.0 - self.cmu) * 0.25 * self.mueff /
                            ((self.n_params + 2) ** 1.5 + 2.0 * self.mueff))

        self.invsqrtC = inv_sqrt(self.cov)[0]
        self.eigen_decomp_updated = self.it

    def _sample(self, n_samples):
        samples = self.random_state.multivariate_normal(
            self.mean, self.var * self.cov, size=n_samples)
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

        if self.log_to_stdout or self.log_to_file:
            self.logger.info("Iteration #%d, fitness: %g"
                             % (self.it, self.fitness[k]))
            self.logger.info("Variance %g" % self.var)

        if (self.it - self.initial_it) % self.n_samples_per_update == 0:
            self._update(self.samples, self.fitness, self.it)

    def _update(self, samples, fitness, it):
        # 1) Update sample distribution mean

        self.last_mean = self.mean
        ranking = np.argsort(fitness, axis=0)
        update_samples = samples[ranking[:self.mu]]
        self.mean = np.sum(self.weights[:, np.newaxis] * update_samples, axis=0)

        mean_diff = self.mean - self.last_mean
        sigma = np.sqrt(self.var)

        # 2) Cumulation: update evolution paths

        # Isotropic (step size) evolution path
        self.ps += (-self.cs * self.ps + self.ps_update_weight / sigma *
                    self.invsqrtC.dot(mean_diff))
        # Anisotropic (covariance) evolution path
        ps_norm_2 = np.linalg.norm(self.ps) ** 2  # Temporary constant
        generation = it / self.n_samples_per_update
        hsig = int(ps_norm_2 / self.n_params /
                   np.sqrt(1 - (1 - self.cs) ** (2 * generation))
                   < self.hsig_threshold)
        self.pc *= 1 - self.cc
        self.pc += (hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) *
                    mean_diff / sigma)

        # 3) Update sample distribution covariance

        # Rank-1 update
        rank_one_update = np.outer(self.pc, self.pc)

        # Rank-mu update
        noise = (update_samples - self.last_mean) / sigma
        rank_mu_update = noise.T.dot(np.diag(self.weights)).dot(noise)

        # Correct variance loss by hsig
        c1a = self.c1 * (1 - (1 - hsig) * self.cc * (2.0 - self.cc))

        if self.active:
            neg_update = samples[ranking[::-1][:self.mu]]
            neg_update -= self.last_mean
            neg_update /= sigma
            neg_rank_mu_update = neg_update.T.dot(np.diag(self.weights)
                                                  ).dot(neg_update)

            self.cov *= 1.0 - c1a - self.cmu + self.neg_cmu * self.alpha_old
            self.cov += rank_one_update * self.c1
            self.cov += rank_mu_update * (self.cmu + self.neg_cmu *
                                          (1.0 - self.alpha_old))
            self.cov -= neg_rank_mu_update * self.neg_cmu
        else:
            self.cov *= 1.0 - c1a - self.cmu
            self.cov += rank_one_update * self.c1
            self.cov += rank_mu_update * self.cmu

        # NOTE here is a bug: it should be cs / (2 * damps), however, that
        #      breaks unit tests and does not improve results
        log_step_size_update = ((self.cs / self.damps) *
                                (ps_norm_2 / self.n_params - 1))
        # NOTE some implementations of CMA-ES use the denominator
        # np.sqrt(self.n_params) * (1.0 - 1.0 / (4 * self.n_params) +
        #                           1.0 / (21 * self.n_params ** 2))
        # instead of self.n_params, in this case cs / damps is correct
        # Adapt step size with factor <= exp(0.6)
        self.var *= np.exp(np.min((0.6, log_step_size_update))) ** 2

        if it - self.eigen_decomp_updated > self.eigen_update_freq:
            self.invsqrtC = inv_sqrt(self.cov)[0]
            self.eigen_decomp_updated = self.it

        self.samples = self._sample(self.n_samples_per_update)

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
        if not (np.all(np.isfinite(self.invsqrtC)) and
                np.all(np.isfinite(self.cov)) and
                np.all(np.isfinite(self.mean)) and
                np.isfinite(self.var)):
            self.logger.info("Stopping: infs or nans" % self.var)
            return True

        if (self.min_variance is not None and
                np.max(np.diag(self.cov)) * self.var <= self.min_variance):
            self.logger.info("Stopping: %g < min_variance" % self.var)
            return True

        max_dist = np.max(pdist(self.fitness[:, np.newaxis]))
        if max_dist < self.min_fitness_dist:
            self.logger.info("Stopping: %g < min_fitness_dist" % max_dist)
            return True

        cov_diag = np.diag(self.cov)
        if (self.max_condition is not None and
                np.max(cov_diag) > self.max_condition * np.min(cov_diag)):
            self.logger.info("Stopping: %g / %g > max_condition"
                             % (np.max(self.cov), np.min(self.cov)))
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
            return -self.best_fitness
        else:
            return self.best_fitness

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["logger"]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.logger = get_logger(self, self.log_to_file, self.log_to_stdout)


class RestartCMAESOptimizer(CMAESOptimizer):
    """CMA-ES with restarts.

    This will outperform plain CMA-ES on multimodal functions.

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

    active : bool, optional (default: False)
        Active CMA-ES (aCMA-ES) with negative weighted covariance matrix
        update

    bounds : array-like, shape (n_samples, 2), optional (default: None)
        Upper and lower bounds for each parameter.

    maximize : optional, boolean (default: True)
        Maximize return or minimize cost?

    min_variance : float, optional (default: 2 * np.finfo(np.float).eps ** 2)
        Minimum variance before restart

    min_fitness_dist : float, optional (default: 2 * np.finfo(np.float).eps)
        Minimum distance between fitness values before restart

    max_condition : float optional (default: 1e7)
        Maximum condition of covariance matrix

    log_to_file: optional, boolean or string (default: False)
        Log results to given file, it will be located in the $BL_LOG_PATH

    log_to_stdout: optional, boolean (default: False)
        Log to standard output

    random_state : optional, int
        Seed for the random number generator.
    """
    def __init__(
            self, initial_params=None, variance=1.0, covariance=None,
            n_samples_per_update=None, active=False, bounds=None,
            maximize=True, min_variance=2 * np.finfo(np.float).eps ** 2,
            min_fitness_dist=2 * np.finfo(np.float).eps, max_condition=1e7,
            log_to_file=False, log_to_stdout=False, random_state=None):
        super(RestartCMAESOptimizer, self).__init__(
            initial_params, variance, covariance, n_samples_per_update,
            active, bounds, maximize, min_variance, min_fitness_dist,
            max_condition, log_to_file, log_to_stdout, random_state)

    def _update(self, samples, fitness, it):
        super(RestartCMAESOptimizer, self)._update(samples, fitness, it)
        if self._test_restart():
            self._reinit()

    def _test_restart(self):
        return super(RestartCMAESOptimizer, self).is_behavior_learning_done()

    def is_behavior_learning_done(self):
        """Returns false because we will restart and not stop.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """
        return False


class IPOPCMAESOptimizer(RestartCMAESOptimizer):
    """Increasing population size CMA-ES.

    After each restart, the population size will be doubled. Hence, the
    solution will be searched more globally.

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

    active : bool, optional (default: False)
        Active CMA-ES (aCMA-ES) with negative weighted covariance matrix
        update

    bounds : array-like, shape (n_samples, 2), optional (default: None)
        Upper and lower bounds for each parameter.

    maximize : optional, boolean (default: True)
        Maximize return or minimize cost?

    min_variance : float, optional (default: 2 * np.finfo(np.float).eps ** 2)
        Minimum variance before restart

    min_fitness_dist : float, optional (default: 2 * np.finfo(np.float).eps)
        Minimum distance between fitness values before restart

    max_condition : float optional (default: 1e7)
        Maximum condition of covariance matrix

    log_to_file: optional, boolean or string (default: False)
        Log results to given file, it will be located in the $BL_LOG_PATH

    log_to_stdout: optional, boolean (default: False)
        Log to standard output

    random_state : optional, int
        Seed for the random number generator.
    """
    def __init__(self, initial_params=None, variance=1.0, covariance=None,
                 n_samples_per_update=None, active=False, bounds=None,
                 maximize=True, min_variance=2 * np.finfo(np.float).eps ** 2,
                 min_fitness_dist=2 * np.finfo(np.float).eps,
                 max_condition=1e7, log_to_file=False, log_to_stdout=False,
                 random_state=None):
        super(IPOPCMAESOptimizer, self).__init__(
            initial_params, variance, covariance, n_samples_per_update,
            active, bounds, maximize, min_variance, min_fitness_dist,
            max_condition, log_to_file, log_to_stdout, random_state)

    def _update(self, samples, fitness, it):
        super(RestartCMAESOptimizer, self)._update(samples, fitness, it)
        if self._test_restart():
            self.n_samples_per_update *= 2
            self._reinit()


class BIPOPCMAESOptimizer(RestartCMAESOptimizer):
    """BI-population CMA-ES.

    After each restart, the population size will be increased or decreased.
    For details, see `the paper
    <http://hal.archives-ouvertes.fr/docs/00/38/20/93/PDF/hansen2009bbi.pdf>`_.

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

    active : bool, optional (default: False)
        Active CMA-ES (aCMA-ES) with negative weighted covariance matrix
        update

    bounds : array-like, shape (n_samples, 2), optional (default: None)
        Upper and lower bounds for each parameter.

    maximize : optional, boolean (default: True)
        Maximize return or minimize cost?

    min_variance : float, optional (default: 2 * np.finfo(np.float).eps ** 2)
        Minimum variance before restart

    min_fitness_dist : float, optional (default: 2 * np.finfo(np.float).eps)
        Minimum distance between fitness values before restart

    max_condition : float optional (default: 1e7)
        Maximum condition of covariance matrix

    log_to_file: optional, boolean or string (default: False)
        Log results to given file, it will be located in the $BL_LOG_PATH

    log_to_stdout: optional, boolean (default: False)
        Log to standard output

    random_state : optional, int
        Seed for the random number generator.
    """
    def __init__(self, initial_params=None, variance=1.0, covariance=None,
                 n_samples_per_update=None, active=False, bounds=None,
                 maximize=True, min_variance=2 * np.finfo(np.float).eps ** 2,
                 min_fitness_dist=2 * np.finfo(np.float).eps,
                 max_condition=1e7, log_to_file=False, log_to_stdout=False,
                 random_state=None):
        super(BIPOPCMAESOptimizer, self).__init__(
            initial_params, variance, covariance, n_samples_per_update,
            active, bounds, maximize, min_variance, min_fitness_dist,
            max_condition, log_to_file, log_to_stdout, random_state)
        self.variance_default = variance
        self.n_iter_large = 0
        self.n_iter_small = 0
        self.n_restarts = 0
        self.large_regime = None

    def _update(self, samples, fitness, it):
        super(RestartCMAESOptimizer, self)._update(samples, fitness, it)
        if self._test_restart():
            if self.n_restarts == 0:
                self.n_samples_default = self.n_samples_per_update

            self.n_restarts += 1

            # Compute budget of the two regimes
            if self.large_regime is None:
                pass
            elif self.large_regime:
                self.n_iter_large += self.it - self.initial_it
            else:
                self.n_iter_small += self.it - self.initial_it

            # Set population size and initial variance for the regime
            if self.n_iter_large > self.n_iter_small:
                self.large_regime = False
                self.n_samples_per_update = int(
                    self.n_samples_default * (0.5 * self.n_samples_large /
                    float(self.n_samples_default)) **
                    (self.random_state.rand() ** 2))
                self.variance = (self.variance_default *
                                 10 ** (-4 * self.random_state.rand()))
            else:
                self.large_regime = True
                self.n_samples_large = (self.n_samples_default *
                                        2 ** self.n_restarts)
                self.n_samples_per_update = self.n_samples_large
                self.variance = self.variance_default

            self._reinit()


cma_types = {"standard": CMAESOptimizer,
             "restart": RestartCMAESOptimizer,
             "ipop": IPOPCMAESOptimizer,
             "bipop": BIPOPCMAESOptimizer}


def fmin(objective_function, cma_type="standard", x0=None,
         eval_initial_x=False, maxfun=1000, maximize=False, *args, **kwargs):
    """Functional interface to the stochastic optimizer CMA-ES.

    Parameters
    ----------
    objective_function : callable
        Objective function

    cma_type : string, optional (default: 'standard')
        Must be one of ['standard', 'restart', 'ipop', 'bipop']

    x0 : array-like, shape = (n_params,), optional (default: 0)
        Initial parameter vector.

    eval_initial_x : bool, optional (default: False)
        Whether the initial parameter vector x0 is evaluated

    maxfun : int, optional (default: 1000)
        The maximum number of function evaluations after which CMA-ES terminates

    maximize : bool, optional (default: False)
        Maximize objective function

    Returns
    -------
    params : array, shape (n_params,)
        Best parameters

    fitness : float
        Fitness value of best parameters
    """
    if cma_type not in cma_types:
        raise ValueError("Unknown cma_type %s. Must be one of %s."
                         % (cma_type, cma_types.keys()))
    else:
        cmaes = cma_types[cma_type](initial_params=x0, maximize=maximize, *args,
                                    **kwargs)

    cmaes.init(x0.shape[0])

    params = np.empty_like(x0)
    for _ in range(maxfun):
        cmaes.get_next_parameters(params)
        cmaes.set_evaluation_feedback(objective_function(params))

    best = (cmaes.get_best_parameters(method="best"), cmaes.get_best_fitness())

    if eval_initial_x:
        f0 = objective_function(x0)
        if maximize and f0 > best[1]:
            best = (np.copy(x0), f0)
        elif not maximize and f0 < best[1]:
            best = (np.copy(x0), f0)

    return best
