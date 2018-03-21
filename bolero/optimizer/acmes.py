# Author: Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import warnings
import numpy as np
from . import Optimizer, CMAESOptimizer
from ..utils.validation import check_random_state, check_feedback
from ..utils.log import get_logger
from ..utils.ranking_svm import RankingSVM


class ACMESOptimizer(Optimizer):
    """CMA-ES with ranking SVM as surrogate model.

    CMA-ES with comparison-based surrogate model (ranking SVM).

    For details, see

    * `presentation <http://loshchilov.com/publications/JET2011_ACMES.pdf>`_
    * `paper <http://loshchilov.com/publications/PPSN2010_ACM-ES.pdf>`_ [1]_

    The authors of the algorithm were creative with the name: the letters of
    ACM-ES have the same meaning as in CMA-ES but the order of the first three
    letters is alphabetically to indicate that it uses a comparison-based
    surrogate model.

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

    n_pre_samples_per_update : int, optional (default: 500)
        Number of samples that will be compared with the surrogate model
        before we select the individuals for a generation

    active : bool, optional (default: False)
        Active CMA-ES (aCMA-ES) with negative weighted covariance matrix
        update

    n_start_iter : int, optional (default: 100)
        Number of iterations before we use the surrogate model

    n_train_max : int, optional (default: 40 + int(4 * n_params ** 1.7))
        Maximum number of training samples for surrogate model

    n_iter_per_sample : int, optional (default: 50000)
        Number of iterations per sample for the quadratic programming solver
        for surrogate model

    bounds : array-like, shape (n_samples, 2), optional (default: None)
        Upper and lower bounds for each parameter.

    maximize : optional, boolean (default: True)
        Maximize return or minimize cost?

    log_to_file: optional, boolean or string (default: False)
        Log results to given file, it will be located in the $BL_LOG_PATH

    log_to_stdout: optional, boolean (default: False)
        Log to standard output

    random_state : int or RandomState, optional (default: None)
        Seed for the random number generator or RandomState object.

    References
    ----------
    .. [1] Loshchilov, I.; Schoenauer, M.; Sebag, M.
        Comparison-Based Optimizers Need Comparison-Based Surrogates,
        Parallel Problem Solving from Nature, 2010.
    """
    def __init__(self, initial_params=None, variance=1.0, covariance=None,
                 n_samples_per_update=None, n_pre_samples_per_update=500,
                 active=False, n_start_iter=100, n_train_max=None,
                 n_iter_per_sample=1000, bounds=None, maximize=True,
                 log_to_file=False, log_to_stdout=False, random_state=None):
        self.initial_params = initial_params
        self.variance = variance
        self.covariance = covariance
        self.n_samples_per_update = n_samples_per_update
        self.n_pre_samples_per_update = n_pre_samples_per_update
        self.active = active
        self.n_start_iter = n_start_iter
        self.n_train_max = n_train_max
        self.n_iter_per_sample = n_iter_per_sample
        self.bounds = bounds
        self.maximize = maximize
        self.log_to_file = log_to_file
        self.log_to_stdout = log_to_stdout
        self.random_state = random_state

    def init(self, dimension):
        if self.n_pre_samples_per_update <= 0:
            raise ValueError("At least one sample must be evaluated by the "
                             "surrogate model. Otherwise you can use standard "
                             "CMA-ES.")
        self.logger = get_logger(self, self.log_to_file, self.log_to_stdout)

        self.random_state = check_random_state(self.random_state)

        self.n_params = dimension
        self.it = 0

        self._reinit()

        self.best_fitness = np.inf
        self.best_fitness_it = self.it
        self.best_params = self.cmaes_opt.best_params.copy()

    def _reinit(self):
        self.cmaes_opt = CMAESOptimizer(
            self.initial_params, self.variance, self.covariance,
            self.n_samples_per_update, self.active, self.bounds, False,
            self.log_to_file, self.log_to_stdout, self.random_state)
        self.cmaes_opt.init(self.n_params)
        self.n_samples_per_update = self.cmaes_opt.n_samples_per_update

        self.samples = self.cmaes_opt._sample(self.n_samples_per_update)
        self.fitness = np.empty(self.n_samples_per_update)
        self.k = 0

        self.sigma_sel0 = 0.4
        self.sigma_sel1 = 0.8

        if self.n_train_max is None:
            self.n_train_max = 40 + int(4 * self.n_params ** 1.7)
        if self.n_train_max > 20000:
            warnings.warn("Maximum number of training samples is clipped to "
                          "%d, got %d previously" % (20000, self.n_train_max))
            self.n_train_max = 20000
        self.n_train_samples = self.n_train_max

        # Training data for the surrogate model
        self.sur_X = np.empty((0, self.n_params))
        self.sur_y = np.empty(0)

        # These variables will be filled if the surrogate model is used
        self.all_samples = None
        self.ranks = None

    def get_next_parameters(self, params):
        params[:] = self.samples[self.k]

    def set_evaluation_feedback(self, feedback):
        f = check_feedback(feedback, compute_sum=True)
        if self.maximize:
            f *= -1

        if f <= self.best_fitness:
            self.best_fitness = f
            self.best_fitness_it = self.it
            self.best_params[:] = self.samples[self.k]

        self.fitness[self.k] = f

        self.it += 1
        self.k += 1

        if self.log_to_stdout or self.log_to_stdout:
            self.logger.info("Iteration #%d, fitness: %g"
                             % (self.it, np.sum(feedback)))

        if self.k >= len(self.fitness):
            self._update()
            self.k = 0

    def _update(self):
        if self.all_samples is not None:
            fitness = self._interpolate_missing_fitness()
            self.cmaes_opt._update(self.all_samples, fitness, self.it)
        else:
            self.cmaes_opt._update(self.samples, self.fitness, self.it)

        # Collect training data for surrogate model
        self.sur_X = np.vstack((self.sur_X, self.samples))[-self.n_train_max:]
        self.sur_y = np.hstack((self.sur_y, self.fitness))[-self.n_train_max:]

        if self.it > self.n_start_iter:
            # Make use of rank-based surrogate model
            pre_samples = self.cmaes_opt._sample(self.n_pre_samples_per_update)

            X_train = self._transform_parameters(
                self.sur_X[np.argsort(self.sur_y)], self.cmaes_opt.mean,
                self.cmaes_opt.invsqrtC)
            transformed_samples = self._transform_parameters(
                pre_samples, self.cmaes_opt.mean, self.cmaes_opt.invsqrtC)

            surrogate_model = self._build_surrogate_model(
                X_train, n_iter=self.n_iter_per_sample * self.n_train_samples,
                random_state=self.random_state)
            ranking_values = surrogate_model.predict(transformed_samples)

            # Select generation from pre-samples based on ranking
            ranked_indices = np.argsort(ranking_values)[::-1]
            selected_rank = self._select_from_distribution(
                self.n_pre_samples_per_update, self.n_samples_per_update,
                self.sigma_sel0)
            selected = ranked_indices[selected_rank]
            self.all_samples = pre_samples[selected]

            # We will evaluate only a fraction of all samples on the real
            # objective function and interpolate the other values based on
            # the ranking
            selected_for_real_evaluation = self._select_from_distribution(
                self.n_samples_per_update,
                np.maximum(3, self.n_samples_per_update // 3), self.sigma_sel1)
            self.ranks = selected_for_real_evaluation

            self.samples = self.all_samples[selected_for_real_evaluation]
            self.fitness = np.empty(len(self.samples))
        else:
            self.samples = self.cmaes_opt._sample(self.n_samples_per_update)
            self.fitness = np.empty(self.n_samples_per_update)

    def _interpolate_missing_fitness(self):
        """Interpolate fitness values between observed samples."""
        fitness = np.empty(self.n_samples_per_update)
        for k in range(1, len(self.ranks)):
            # Interpolate between neighbors according to rank
            better = self.ranks[k - 1]
            worse = self.ranks[k]
            fitness[better:worse + 1] = np.linspace(
                self.fitness[k - 1], self.fitness[k],
                worse - better + 1)
        # Fitness becomes exponentially worse at the end
        for k in range(self.ranks[-1] + 1, self.n_samples_per_update):
            fitness[k] = fitness[k - 1] + k * 1e-5 * fitness[k - 1]
        return fitness

    def _build_surrogate_model(self, X_train, epsilon=1.0, c_base=6.0,
                               c_pow=2.0, c_sigma=1.0, n_iter=None,
                               random_state=None):
        """Build surrogate model based on ranking SVM."""
        surrogate_model = RankingSVM(
            n_iter=n_iter, epsilon=epsilon, c_base=c_base, c_pow=c_pow,
            c_sigma=c_sigma, random_state=random_state)
        surrogate_model.fit(X_train)
        return surrogate_model

    def _transform_parameters(self, samples, mean, invsqrtC):
        """Transform parameters based on CMA-ES state."""
        return np.dot(samples - mean, invsqrtC)

    def _select_from_distribution(self, n_pre_samples, n_samples, sigma):
        """Select samples indices."""
        selected = [0]
        while len(selected) < n_samples:
            idx = int(np.clip(sigma * n_pre_samples *
                              np.abs(self.random_state.randn()),
                              0, n_pre_samples - 1))
            if idx not in selected:
                selected.append(idx)
        return np.sort(selected)

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
            return self.cmaes_opt.get_best_parameters(method)

    def is_behavior_learning_done(self):
        return False

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["logger"]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.logger = get_logger(self, self.log_to_file, self.log_to_stdout)
