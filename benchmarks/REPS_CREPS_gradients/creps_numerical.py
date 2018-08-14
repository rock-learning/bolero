# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#          Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from bolero.optimizer import CREPSOptimizer
from bolero.utils.mathext import logsumexp


def solve_dual_contextual_reps(S, R, epsilon, min_eta, approx_grad = True):
    """Solve dual function for C-REPS.

    Parameters
    ----------
    S : array, shape (n_samples_per_update, n_context_features)
        Features for the context-dependend reward baseline

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

    nu : array, shape (n_context_features,)
        Coefficients of linear reward baseline function
    """
    if S.shape[0] != R.shape[0]:
        raise ValueError("Number of contexts (%d) must equal number of "
                         "returns (%d)." % (S.shape[0], R.shape[0]))

    n_samples_per_update = len(R)

    # Definition of the dual function
    def g(x):  # Objective function
        eta = x[0]
        nu = x[1:]
        return (eta * epsilon + nu.T.dot(S.mean(axis=0)) +
                eta * logsumexp((R - nu.dot(S.T)) / eta,
                    b=1.0 / n_samples_per_update))

    # Lower bound for Lagrange parameters eta and nu
    bounds = np.vstack(([[min_eta, None]], np.tile(None, (S.shape[1], 2))))
    # Start point for optimization
    x0 = [1] + [1] * S.shape[1]

    # Perform the actual optimization of the dual function
    #r = NLP(g, x0, lb=lb).solve('ralg', iprint=-10)
    r = fmin_l_bfgs_b(g, x0, approx_grad=True, bounds=bounds)

    # Fetch optimal lagrangian parameter eta. Corresponds to a temperature
    # of a softmax distribution
    eta = r[0][0]
    # Fetch optimal vale of vector nu which determines the context
    # dependent baseline
    nu = r[0][1:]

    # Determine weights of individual samples based on the their return,
    # the optimal baseline nu.dot(\phi(s)) and the "temperature" eta
    log_d = (R - nu.dot(S.T)) / eta
    # Numerically stable softmax version of the weights. Note that
    # this does neither changes the solution of the weighted least
    # squares nor the estimation of the covariance.
    d = np.exp(log_d - log_d.max())
    d /= d.sum()

    return d, eta, nu


class CREPSOptimizerNumerical(CREPSOptimizer):
    """Contextual Relative Entropy Policy Search (using umerical gradients).

    Inherits all parameters and methods from CREPSOptimizer, with the
    only difference being 'set_evaluation_feedback' using a modified
    'solve_dual_contextual_reps' function which uses numerical gradients
    when minimizing the dual function.
    """
    def set_evaluation_feedback(self, rewards):
        """Set feedbacks for the parameter vector.

        Parameters
        ----------
        rewards : list of float
            Feedbacks for each step or for the episode, depends on the problem
        """
        self._add_sample(rewards)

        if self.it % self.train_freq == 0:
            phi_s = np.asarray(self.history_phi_s)
            theta = np.asarray(self.history_theta)
            R = np.asarray(self.history_R)

            self.weights = solve_dual_contextual_reps(
                phi_s, R, self.epsilon, self.min_eta, approx_grad = self.approx_grad)[0]
            # NOTE the context have already been transformed
            self.policy_.fit(phi_s, theta, self.weights,
                             context_transform=False)
