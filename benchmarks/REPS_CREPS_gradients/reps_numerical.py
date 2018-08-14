# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#          Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from bolero.optimizer import REPSOptimizer
from bolero.utils.mathext import logsumexp
from bolero.utils.validation import check_feedback


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
    def g(eta):  # Objective function, no gradient
		return eta * (epsilon + logsumexp(R / eta, b=1.0 / len(R)))

    # Lower bound for Lagrangian eta
    bounds = np.array([[min_eta, None]])
    # Start point of optimization
    x0 = [1]

    # Perform the actual optimization of the dual function
    r = fmin_l_bfgs_b(g, x0, approx_grad=True, bounds=bounds)

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


class REPSOptimizerNumerical(REPSOptimizer):
    """Relative Entropy Policy Search (REPS) (using umerical gradients).

    Inherits all parameters and methods from REPSOptimizer, with the
    only difference being 'set_evaluation_feedback' using a modified
    'solve_dual_reps' function which uses numerical gradients when
    minimizing the dual function.
    """
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
