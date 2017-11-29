# Author: Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import numpy as np
from collections import deque
from ..optimizer import ContextualOptimizer
from ..representation.ul_policies import (ContextTransformationPolicy,
                                          LinearGaussianPolicy)
from ..utils.validation import check_random_state, check_feedback, check_context
from ..utils.log import get_logger
from .cmaes import inv_sqrt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge


class CCMAESOptimizer(ContextualOptimizer):
    """Contextual Covariance Matrix Adaptation Evolution Strategy.

    This contextual version of :class:`~bolero.optimizer.CMAESOptimizer`
    inherits the properties from the original algorithm. More information
    on the algorithm can be found in the original publication [1]_. A major
    advantage over C-REPS is that it quickly adapts the step size.

    C-CMA-ES internally models the context-dependent baseline with polynomial
    ridge regression. The degree of the polynomial can be configured. The
    regularization coefficient is shared with the upper-level policy.
    Context features are only used for the upper-level policy.

    Parameters
    ----------
    initial_params : array-like, shape (n_params,)
        Initial parameter vector.

    variance : float, optional (default: 1.0)
        Initial exploration variance.

    covariance : array-like, optional (default: None)
        Either a diagonal (with shape (n_params,)) or a full covariance matrix
        (with shape (n_params, n_params)). A full covariance can contain
        information about the correlation of variables.

    n_samples_per_update : int, optional
        Number of samples that will be used to update a policy.
        default: 4 + int(3*log(n_params + n_context_dims)) *
                 (1 + 2 * n_context_dims)

    context_features : string or callable, optional (default: None)
        (Nonlinear) feature transformation for the context, which will be used
        to learn the upper-level policy.

    baseline_degree : int, optional (default: 2)
        Degree of the polynomial features that will be used to estimate the
        context-dependent reward baseline.

    gamma : float, optional (default: 1e-4)
        Regularization parameter for baseline and upper-level policy.

    log_to_file: optional, boolean or string (default: False)
        Log results to given file, it will be located in the $BL_LOG_PATH

    log_to_stdout: optional, boolean (default: False)
        Log to standard output

    random_state : optional, int
        Seed for the random number generator.

    References
    ----------
    .. [1] Abdolmaleki, A.; Price, B.; Lau, N.; Paulo Reis, L.; Neumann, G.
        Contextual Covariance Matrix Adaptation Evolution Strategies.
    """
    def __init__(self, initial_params=None, variance=1.0, covariance=None,
                 n_samples_per_update=None, context_features=None,
                 baseline_degree=2, gamma=1e-4, log_to_file=False,
                 log_to_stdout=False, random_state=None, **kwargs):
        self.initial_params = initial_params
        self.variance = variance
        self.covariance = covariance
        self.n_samples_per_update = n_samples_per_update
        self.context_features = context_features
        self.baseline_degree = baseline_degree
        self.gamma = gamma
        self.log_to_file = log_to_file
        self.log_to_stdout = log_to_stdout
        self.random_state = random_state

    def init(self, n_params, n_context_dims):
        """Initialize optimizer.

        Parameters
        ----------
        n_params : int
            number of parameters

        n_context_dims : int
            number of dimensions of the context space
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
        self.n_params = n_params
        if self.covariance is None:
            self.covariance = np.eye(n_params)
        elif self.covariance.ndim == 1:
            self.covariance = np.diag(self.covariance)

        self.context = None
        self.params = None
        self.reward = None

        self.policy_ = ContextTransformationPolicy(
            LinearGaussianPolicy, n_params, n_context_dims,
            context_transformation=self.context_features,
            mean=self.initial_params, covariance_scale=1.0, gamma=self.gamma,
            random_state=self.random_state)
        self.cov = np.copy(self.covariance)
        self.var = self.variance
        self.policy_.policy.Sigma = self.variance * self.covariance

        self.n_total_dims = n_params + n_context_dims
        if self.n_samples_per_update is None:
            self.n_samples_per_update = (
                4 + int(3 * np.log(self.n_total_dims)) *
                (1 + 2 * n_context_dims))

        self.hsig_threshold = 2 + 4.0 / (self.n_params + 1)

        self.history_theta = deque(maxlen=self.n_samples_per_update)
        self.history_R = deque(maxlen=self.n_samples_per_update)
        self.history_s = deque(maxlen=self.n_samples_per_update)
        self.history_phi_s = deque(maxlen=self.n_samples_per_update)

        # Evolution path for covariance
        self.pc = np.zeros(self.n_params)
        # Evolution path for sigma
        self.ps = np.zeros(self.n_params)

        self.mu = self.n_samples_per_update // 2
        self.ordered_weights = np.zeros(self.n_samples_per_update)
        self.ordered_weights[:self.mu] = (
            np.log(self.mu + 0.5) - np.log1p(np.arange(int(self.mu))))
        self.ordered_weights /= np.sum(self.ordered_weights)
        self.mueff = 1.0 / np.sum(self.ordered_weights ** 2)

        self.c1 = 2.0 / ((self.n_total_dims + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1.0 - self.c1,
            2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) /
            ((self.n_total_dims + 2.0) ** 2 + self.mueff))
        self.cc = ((4.0 + self.mueff / self.n_total_dims) /
                   (4.0 + self.n_total_dims +
                    2.0 * self.mueff / self.n_total_dims))

        self.c_sigma = (self.mueff + 2.0) / float(
            self.n_total_dims + self.mueff + 5.0)
        self.d_sigma = ((1.0 + 2.0 * max(
            0.0, np.sqrt((self.mueff - 1.0) /
                         (self.n_total_dims + 1.0)) - 1.0) + self.c_sigma)
            + np.log(n_context_dims + 1.0))
        self.randn_vector_norm = (
            np.sqrt(self.n_params) * (1.0 - 1.0 / (4.0 * self.n_params)
                                      + 1.0 / (21.0 * self.n_params ** 2)))

        self.weights = np.empty(self.n_samples_per_update)

        self.reward_model_features = PolynomialFeatures(
            degree=self.baseline_degree, include_bias=False)
        self.reward_model = Ridge(alpha=self.gamma, fit_intercept=True)

        self.invsqrtC = inv_sqrt(self.cov)[0]
        self.eigen_decomp_updated = self.it
        self.eigen_update_freq = (self.n_samples_per_update /
                                  ((self.c1 + self.cmu) * self.n_params * 10))

    def get_desired_context(self):
        """C-REPS does not actively select the context.

        Returns
        -------
        context : None
            C-REPS does not have any preference
        """
        return None

    def set_context(self, context):
        """Set context of next evaluation.

        Parameters
        ----------
        context : array-like, shape (n_context_dims,)
            The context in which the next rollout will be performed
        """
        self.context = check_context(context)

    def get_next_parameters(self, params, explore=True):
        """Get next individual/parameter vector for evaluation.

        Parameters
        ----------
        params : array_like, shape (n_params,)
            Parameter vector, will be modified

        explore : bool, optional (default: True)
            Whether we want to turn exploration on for the next evaluation
        """
        self.params = self.policy_(self.context, explore=explore)
        params[:] = self.params

    def set_evaluation_feedback(self, rewards):
        """Set feedbacks for the parameter vector.

        Parameters
        ----------
        rewards : list of float
            Feedbacks for each step or for the episode, depends on the problem
        """
        self._add_sample(rewards)

        if self.it % self.n_samples_per_update == 0:
            self._update()

    def _add_sample(self, rewards):
        self.reward = check_feedback(rewards, compute_sum=True)
        self.logger.info("[CCMAES] Reward %.6f" % self.reward)

        phi_s = self.policy_.transform_context(self.context)

        self.history_theta.append(self.params)
        self.history_R.append(self.reward)
        self.history_s.append(self.context)
        self.history_phi_s.append(phi_s)

        self.it += 1

    def _update(self):
        s = np.asarray(self.history_s)
        phi_s = np.asarray(self.history_phi_s)
        theta = np.asarray(self.history_theta)
        R = np.asarray(self.history_R)

        baseline_features = self.reward_model_features.fit_transform(s)
        self.reward_model.fit(baseline_features, R)
        advantages = R - self.reward_model.predict(baseline_features)
        indices = np.argsort(np.argsort(advantages)[::-1])

        self.weights = self.ordered_weights[indices]

        if self.it - self.eigen_decomp_updated > self.eigen_update_freq:
            self.invsqrtC = inv_sqrt(self.cov)[0]
            self.eigen_decomp_updated = self.it

        mean_phi = np.mean(phi_s, axis=0)
        last_mean_theta = self.policy_.policy(mean_phi, explore=False)
        last_mean_thetas = phi_s.dot(self.policy_.W.T)

        self.policy_.fit(phi_s, theta, self.weights, context_transform=False)

        mean_theta = self.policy_.policy(mean_phi, explore=False)
        sigma = np.sqrt(self.var)
        mean_diff = (mean_theta - last_mean_theta) / sigma

        self.ps *= (1.0 - self.c_sigma)

        self.ps += (np.sqrt(self.c_sigma * (2.0 - self.c_sigma) *
                            self.mueff) * self.invsqrtC.dot(mean_diff))

        ps_norm_2 = np.linalg.norm(self.ps) ** 2  # Temporary constant
        generation = self.it / self.n_samples_per_update
        hsig = int(ps_norm_2 / self.n_params /
                np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * generation))
                < self.hsig_threshold)
        self.pc *= 1.0 - self.cc
        self.pc += (hsig * np.sqrt(self.cc * (2.0 - self.cc) * self.mueff) *
                    mean_diff)

        # Rank-1 update
        rank_one_update = np.outer(self.pc, self.pc)

        # Rank-mu update
        noise = (theta - last_mean_thetas) / sigma
        rank_mu_update = noise.T.dot(np.diag(self.weights)).dot(noise)

        # Correct variance loss by hsig
        c1a = self.c1 * (1 - (1 - hsig) * self.cc * (2.0 - self.cc))

        self.cov *= (1.0 - c1a - self.cmu)
        self.cov += self.cmu * rank_mu_update
        self.cov += self.c1 * rank_one_update

        log_step_size_update = (
            (self.c_sigma / self.d_sigma) *
            (np.sqrt(ps_norm_2) / self.randn_vector_norm - 1.0))
        # Adapt step size with factor <= exp(0.6)
        self.var *= np.exp(np.min((0.6, log_step_size_update))) ** 2
        self.policy_.policy.Sigma = self.var * self.cov

    def best_policy(self):
        """Return current best estimate of contextual policy.

        Returns
        -------
        policy : UpperLevelPolicy
            Best estimate of upper-level policy
        """
        return self.policy_

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
