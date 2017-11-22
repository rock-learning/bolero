import numpy as np
try:
    from skopt.optimizer import Optimizer as _SkOptOptimizer
    from skopt.learning import (ExtraTreesRegressor, RandomForestRegressor,
                                GaussianProcessRegressor,
                                GradientBoostingQuantileRegressor)
    from skopt.learning.gaussian_process.kernels import ConstantKernel
    from skopt.learning.gaussian_process.kernels import Matern
    skopt_available = True
except:
    skopt_available = False
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import check_random_state
from .optimizer import Optimizer
from ..utils.validation import check_feedback
from ..utils.log import get_logger


class SkOptOptimizer(Optimizer):
    """Bayesian Optimization from scikit-optimize.

    See the `project website <https://github.com/scikit-optimize/scikit-optimize>`_
    for details.

    Parameters
    ----------
    dimensions : list, shape=(n_dims,)
        List of search space dimensions.
        Each search dimension can be defined either as

        - a `(upper_bound, lower_bound)` tuple (for `Real` or `Integer`
          dimensions),
        - a `(upper_bound, lower_bound, "prior")` tuple (for `Real`
          dimensions),
        - as a list of categories (for `Categorical` dimensions), or
        - an instance of a `Dimension` object (`Real`, `Integer` or
          `Categorical`).

    base_estimator : string or Regressor, optional (default: 'ET')
        The regressor to use as surrogate model. Can be either

        - `"RF"` for random forest regressor
        - `"ET"` for extra trees regressor
        - `"GP"` for Gaussian process estimator with Matern kernel
        - `"GBRT"` for gradient boosted trees
        - instance of regressor with support for `return_std` in its predict
          method

        The predefined models are initilized with good defaults. If you
        want to adjust the model parameters pass your own instance of
        a regressor which returns the mean and standard deviation when
        making predictions.

    maximize : boolean, optional (default: True)
        Maximize return or minimize cost?

    n_random_starts : int, optional (default: 10)
        Number of evaluations of `func` with random initialization points
        before approximating the `func` with `base_estimator`. While random
        points are being suggested no model will be fit to the observations.

    acq_func : string, optional (default='EI')
        Function to minimize over the posterior distribution. Can be either

        - `"LCB"` for lower confidence bound,
        - `"EI"` for negative expected improvement,
        - `"PI"` for negative probability of improvement.

    acq_optimizer : string, 'sampling' or 'lbfgs', optional (default:'lbfgs')
        Method to minimize the acquistion function. The fit model
        is updated with the optimal value obtained by optimizing `acq_func`
        with `acq_optimizer`.

        - If set to `"sampling"`, then `acq_func` is optimized by computing
          `acq_func` at `n_points` sampled randomly.
        - If set to `"lbfgs"`, then `acq_func` is optimized by
              - Sampling `n_restarts_optimizer` points randomly.
              - `"lbfgs"` is run for 20 iterations with these points as initial
                points to find local minima.
              - The optimal of these local minima is used to update the prior.

    random_state : int, RandomState instance, or None, optional (default: None)
        Set random state to something other than None for reproducible
        results.

    n_points : int, optional (default: 500)
        Number of points to sample to determine the next "best" point.
        Useless if acq_optimizer is set to `"lbfgs"`.

    n_restarts_optimizer : int, optional (default: 5)
        The number of restarts of the optimizer when `acq_optimizer`
        is `"lbfgs"`.

    xi : float, optional (default: 0.01)
        Controls how much improvement one wants over the previous best
        values. Used when the acquisition is either `"EI"` or `"PI"`.

    kappa : float, optional (default: 1.96)
        Controls how much of the variance in the predicted values should be
        taken into account. If set to be very high, then we are favouring
        exploration over exploitation and vice versa.
        Used when the acquisition is `"LCB"`.

    n_jobs : int, optional (default: 1)
        Number of cores to run in parallel while running the lbfgs
        optimizations over the acquisition function. Valid only when
        `acq_optimizer` is set to "lbfgs."
        Defaults to 1 core. If `n_jobs=-1`, then number of jobs is set
        to number of cores.
    """
    def __init__(self, dimensions, base_estimator="GP", maximize=True,
                 n_random_starts=10, acq_func="LCB", acq_optimizer="lbfgs",
                 random_state=None, n_points=10000, n_restarts_optimizer=5,
                 xi=0.01, kappa=1.96, n_jobs=1):
        if not skopt_available:
            raise ImportError("skopt is not installed correctly")
        self.maximize = maximize
        self.n_params = len(dimensions)

        rng = check_random_state(random_state)
        if isinstance(base_estimator, str):
            if base_estimator == "RF":
                base_estimator = RandomForestRegressor(
                    n_estimators=100, min_samples_leaf=3, n_jobs=n_jobs,
                    random_state=rng)
            elif base_estimator == "ET":
                base_estimator = ExtraTreesRegressor(
                    n_estimators=100, min_samples_leaf=3, n_jobs=n_jobs,
                    random_state=rng)
            elif base_estimator == "GP":
                cov_amplitude = ConstantKernel(1.0, (0.01, 1000.0))
                matern = Matern(
                    length_scale=np.ones(len(dimensions)),
                    length_scale_bounds=[(0.01, 100)] * len(dimensions),
                    nu=2.5)
                base_estimator = GaussianProcessRegressor(
                    kernel=cov_amplitude * matern,
                    normalize_y=True, random_state=rng, alpha=0.0,
                    noise="gaussian", n_restarts_optimizer=2)
            elif base_estimator == "GBRT":
                gbrt = GradientBoostingRegressor(
                    n_estimators=30, loss="quantile")
                base_estimator = GradientBoostingQuantileRegressor(
                    base_estimator=gbrt, n_jobs=n_jobs, random_state=rng)
            else:
                raise ValueError(
                    "Valid strings for the base_estimator parameter"
                    " are: 'RF', 'ET', or 'GP', not '%s'" % base_estimator)

        acq_func_kwargs = {
            "xi": xi,
            "kappa": kappa
        }
        acq_optimizer_kwargs = {
            "n_points": n_points,
            "n_restarts_optimizer": n_restarts_optimizer,
            "n_jobs": n_jobs
        }
        self.optimizer = _SkOptOptimizer(
            dimensions=dimensions, base_estimator=base_estimator,
            n_initial_points=n_random_starts, acq_func=acq_func,
            acq_optimizer=acq_optimizer, random_state=random_state,
            acq_func_kwargs=acq_func_kwargs,
            acq_optimizer_kwargs=acq_optimizer_kwargs)

    def init(self, n_params):
        """Initialize the behavior search.

        Parameters
        ----------
        n_params : int
            dimension of the parameter vector
        """
        if self.n_params != n_params:
            raise ValueError("Number of dimensions (%d) does not match "
                             "number of given parameter specifications (%d)."
                             % (n_params, self.n_params))
        self.current_params = None
        self.best_fitness = np.inf
        self.best_params = None

    def get_next_parameters(self, params):
        """Get next individual/parameter vector for evaluation.

        Parameters
        ----------
        params : array_like, shape (n_params,)
            Parameter vector, will be modified
        """
        self.current_params = self.optimizer.ask()
        params[:] = self.current_params

    def set_evaluation_feedback(self, feedback):
        """Set feedbacks for the parameter vector.

        Parameters
        ----------
        feedback : list of float
            feedbacks for each step or for the episode, depends on the problem
        """
        feedback = check_feedback(feedback, compute_sum=True)
        if self.maximize:
            feedback *= -1.0
        if feedback < self.best_fitness:
            self.best_fitness = feedback
            self.best_params = np.copy(self.current_params)
        self.optimizer.tell(self.current_params, feedback)

    def is_behavior_learning_done(self):
        """Check if the optimization is finished.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """
        return False

    def get_best_parameters(self):
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
        return self.best_params

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
