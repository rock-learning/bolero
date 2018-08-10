# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#          Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from collections import deque
from .optimizer import Optimizer
from ..utils.scaling import Scaling
from ..utils.mathext import logsumexp
from ..representation.ul_policies import BoundedScalingPolicy
from ..representation.ul_policies import ConstantGaussianPolicy
from ..utils.validation import check_random_state, check_feedback
from ..utils.log import get_logger


def solve_dual_reps(R, epsilon, min_eta):
    """Solve dual function for REPS.

    Parameters
    ----------
    R : array, shape (n_samples_per_update,)
        Corresponding obtained rewards

    epsilon : float
        Maximum Kullback-Leibler divergence of two successive policy
        distributions.

    min_eta : float
        Minimum eta, 0 would result in numerical problems

    Returns
    -------
    d : array, shape (n_samples_per_update,)
        Weights for training samples

    eta : float
        Temperature
    """
    if R.ndim != 1:
        raise ValueError("Returns must be passed in a flat array!")

    R_max = R.max()
    R_min = R.min()
    if R_max == R_min:
        return np.ones(R.shape[0]) / float(R.shape[0]), np.nan  # eta not known
    # Normalize returns into range [0, 1] such that eta (and min_eta)
    # always lives on the same scale
    R = (R - R_min) / (R_max - R_min)

    # Definition of the dual function
    def g(eta):  # Objective function
        R_over_eta = R / eta
        Z = np.exp(R_over_eta - R_over_eta.max())
        log_sum_exp = logsumexp(R_over_eta, b=1.0 / len(R))

        f = eta * (epsilon + log_sum_exp)
        d_eta = epsilon + log_sum_exp - Z.dot(R_over_eta) / Z.sum()
        return f, np.array([d_eta])

    # Lower bound for Lagrangian eta
    bounds = np.array([[min_eta, None]])
    # Start point of optimization
    x0 = [1]

    # Perform the actual optimization of the dual function
    r = fmin_l_bfgs_b(g, x0, bounds=bounds)

    # Fetch optimal Lagrangian parameter eta. Corresponds to a temperature
    # of a softmax distribution
    eta = r[0][0]

    # Determine weights of individual samples based on the their return and
    # the "temperature" eta
    log_d = R / eta
    # Numerically stable softmax version of the weights. Note that
    # this does neither changes the solution of the weighted least
    # squares nor the estimation of the covariance.
    d = np.exp(log_d - log_d.max())
    d /= d.sum()

    return d, r[0]


class REPSOptimizer(Optimizer):
    """Relative Entropy Policy Search (REPS) as Optimizer.

    Use REPS as a black-box optimizer: learn an upper-level distribution
    :math:`\pi(\\boldsymbol{\\theta})` which selects weights
    :math:`\\boldsymbol{\\theta}` for the objective function. At the moment,
    :math:`\pi(\\boldsymbol{\\theta})` is assumed to be a multivariate
    gaussian distribution whose mean and covariance (governing exploration)
    are learned. REPS constrains the learning updates such that the KL
    divergence between the old and the new distribution is below a threshold
    epsilon. More details can be found in the original publication [1]_.

    Abdolmaleki et al. [2]_ state that
    "the episodic REPS algorithm uses a sample based approximation of the
    KL-bound, which needs a lot of samples in order to be accurate. Moreover,
    a typical problem of REPS is that the entropy of the search distribution
    decreases too quickly, resulting in premature convergence."

    Parameters
    ----------
    initial_params : array, shape = (num_params,), optional (default: zeros)
        Initial parameter vector.

    variance : float, optional (default: 1)
        Initial exploration variance.

    covariance : array-like, optional (default: None), optional (default: I)
        Either a diagonal (with shape (n_params,)) or a full covariance matrix
        (with shape (n_params, n_params)). A full covariance can contain
        information about the correlation of variables.

    epsilon : float > 0.0, optional (default: 2)
        The maximum the KL divergence between old and new "data" distribution
        might take on

    train_freq : int > 0, optional (default: 25)
        The frequency (the number of rollouts) of training, i.e., using REPS
        for updating the policies parameters. Defaults to 25 rollouts.

    min_eta : float, optional (default: 1e-8)
        Minimum eta, 0 would result in numerical problems

    n_samples_per_update : int, optional (default: 100)
        Number of samples that will be used to update a policy.

    bounds : array-like, shape (n_samples, 2), optional (default: None)
        Upper and lower bounds for each parameter.

    log_to_file : optional, boolean or string (default: False)
        Log results to given file, it will be located in the $BL_LOG_PATH

    log_to_stdout : optional, boolean (default: False)
        Log to standard output

    random_state : optional, int
        Seed for the random number generator.

    References
    ----------
    .. [1] Peters, J.; Muelling, K.; Altuen, Y. Relative Entropy Policy Search.
        Proceedings of the Twenty-Fourth AAAI Conference on Artificial
        Intelligence, 2010.

    .. [2] Abdolmaleki, A.; Lioutikov, R.; Lau, N; Paulo Reis, L.; Peters, J.;
        Neumann, G. Model-Based Relative Entropy Stochastic Search.
        Advances in Neural Information Processing Systems 28, 2015.
    """
    def __init__(self, initial_params=None, variance=1.0, covariance=None,
                 epsilon=2.0, min_eta=1e-8, train_freq=25,
                 n_samples_per_update=100, bounds=None, log_to_file=False,
                 log_to_stdout=False, random_state=None):
        self.initial_params = initial_params
        self.variance = variance
        self.covariance = covariance
        self.epsilon = epsilon
        self.min_eta = min_eta
        self.train_freq = train_freq
        self.n_samples_per_update = n_samples_per_update
        self.bounds = bounds
        self.log_to_file = log_to_file
        self.log_to_stdout = log_to_stdout
        self.random_state = random_state

    def init(self, n_params):
        """Initialize optimizer.

        Parameters
        ----------
        n_params : int
            number of parameters
        """
        self.logger = get_logger(self, self.log_to_file, self.log_to_stdout)

        self.random_state = check_random_state(self.random_state)

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

        self.params = None
        self.reward = None

        scaling = Scaling(variance=self.variance, covariance=self.covariance,
                          compute_inverse=True)
        self.policy_ = BoundedScalingPolicy(ConstantGaussianPolicy(
            n_params, mean=scaling.inv_scale(self.initial_params),
            random_state=self.random_state), scaling=scaling,
            bounds=self.bounds)

        # Maximum return obtained
        self.max_return = -np.inf
        # Best parameters found so far
        self.best_params = self.initial_params.copy()

        self.history_theta = deque(maxlen=self.n_samples_per_update)
        self.history_R = deque(maxlen=self.n_samples_per_update)

    def get_next_parameters(self, params, explore=True):
        """Return parameter vector that shall be evaluated next.

        Parameters
        ----------
        params : array-like, shape = (n_params,)
            The selected parameters will be written into this as a side-effect.

        explore : bool
            Whether exploration in parameter selection is enabled
        """
        self.params = self.policy_(None, explore=explore)
        params[:] = self.params

    def set_evaluation_feedback(self, feedbacks):
        """Inform optimizer of outcome of a rollout with current weights."""
        self.reward = check_feedback(feedbacks, compute_sum=True)

        self.history_theta.append(self.params)
        self.history_R.append(self.reward)

        self.it += 1

        if self.it % self.train_freq == 0:
            theta = np.asarray(self.history_theta)
            R = np.asarray(self.history_R)
            d = solve_dual_reps(R, self.epsilon, self.min_eta)[0]
            self.policy_.fit(None, theta, d)

        self.logger.info("Reward %.6f" % self.reward)

        if self.reward > self.max_return:
            self.max_return = self.reward
            self.best_params = self.params

    def get_best_parameters(self):
        """Get the best parameters.

        Returns
        -------
        best_params : array-like, shape (n_params,)
            Best parameters
        """
        return self.best_params

    def is_behavior_learning_done(self):
        """Check if the optimization is finished.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """
        return False

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["logger"]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.logger = get_logger(self, self.log_to_file, self.log_to_stdout)
