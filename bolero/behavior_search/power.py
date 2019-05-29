# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#         Alexander Fabisch <alexander.fabisch@dfki.de>

import numpy as np
import heapq
import dmp
from .behavior_search import BehaviorSearch, PickableMixin
from ..utils.validation import check_random_state, check_feedback
from ..utils.log import get_logger


class PoWERWithDMP(PickableMixin, BehaviorSearch):
    """Policy learning by Weighting Explorations with the Returns (PoWER).

    This version of PoWER uses a DMP as policy.

    Paper available from
    `NeurIPS <https://papers.nips.cc/paper/3545-policy-search-for-motor-primitives-in-robotics.pdf>`_.
    Based on the Matlab code of Kober et al.: `source
    <http://www.ias.informatik.tu-darmstadt.de/uploads/Member/JensKober/matlab_PoWER.zip>`_.

    Parameters
    ----------
    initial_params : array-like, shape = (n_params,), optional (default: 0s)
        Initial parameter vector.

    variance : float, optional (default: 1.0)
        Initial exploration variance.

    covariance : array-like, optional (default: None)
        A diagonal (with shape (n_params,)) covariance matrix.

    n_samples_per_update : integer, optional (default: 10)
        Number of roll-outs that are required for a parameter update.

    reward_transformation : callable, optional (default: identity)
        A function that transforms the rewards (usually to the interval [0, 1],
        where 1 is best best possible value). PoWER requires the reward
        function be an improper probability distribution, i.e. all rewards must
        be positive. It can also be a proper probability distribution, i.e. sum
        up to one (during an episode?), which will be beneficial for the
        learning speed. An example for a reward transformation function is
        lambda r: numpy.exp(s*r), where s is a scaling factor and r is the
        reward.

    log_to_file: boolean or string, optional (default: False)
        Log results to given file, it will be located in the $BL_LOG_PATH

    log_to_stdout: boolean, optional (default: False)
        Log to standard output

    random_state : int or RandomState, optional (default: None)
        Seed for the random number generator or RandomState object.
    """
    def __init__(self, dmp_behavior, variance=1.0, covariance=None,
                 n_samples_per_update=10, reward_transformation=lambda r: r,
                 log_to_file=False, log_to_stdout=False, random_state=None):
        self.dmp_behavior = dmp_behavior
        self.variance = variance
        self.covariance = covariance
        self.n_samples_per_update = n_samples_per_update
        self.reward_transformation = reward_transformation
        self.log_to_file = log_to_file
        self.log_to_stdout = log_to_stdout
        self.random_state = random_state

    def init(self, n_inputs, n_outputs):
        """Initialize the behavior search.

        Parameters
        ----------
        n_inputs : int
            number of inputs of the behavior

        n_outputs : int
            number of outputs of the behavior
        """
        self.logger = get_logger(self, self.log_to_file, self.log_to_stdout)
        self.random_state = check_random_state(self.random_state)

        self.dmp_behavior.init(n_inputs, n_outputs)
        self.mean = self.dmp_behavior.get_params()
        self.n_params = len(self.mean)

        # Entries of best_rollouts have the form:
        # (return, random-value, parameters, Q values, exploration variance)
        self.best_rollouts = []

        time = np.arange(
            0.0, self.dmp_behavior.execution_time + self.dmp_behavior.dt,
            self.dmp_behavior.dt)
        phases = [dmp.phase(t, self.dmp_behavior.alpha_z,
                            self.dmp_behavior.execution_time, 0.0)
                  for t in time]
        self.basis = np.array([self._dmp_activations(z) for z in phases])
        """
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(time, self.basis)
        plt.show()
        #"""
        self.tmp_outer = np.array([
            np.outer(self.basis[i], self.basis[i])
            for i in range(self.basis.shape[0])])

        if self.covariance is None:
            self.cov = np.ones(self.n_params)
        else:
            self.cov = np.asarray(self.covariance).copy()
        self.cov *= self.variance

        self.it = 0

    # TODO this is copied from the DMP implementation...
    def _dmp_activations(self, z):
        activations = np.exp(
            -self.dmp_behavior.widths * (z - self.dmp_behavior.centers) ** 2)
        activations /= activations.sum()
        return activations

    def get_next_behavior(self):
        """Obtain next behavior for evaluation.

        Returns
        -------
        behavior : Behavior
            mapping from input to output
        """
        self.noise = np.sqrt(self.cov) * self.random_state.randn(self.n_params)
        self.params = self.mean + self.noise
        self.dmp_behavior.set_params(self.params)
        self.dmp_behavior.reset()
        return self.dmp_behavior

    def set_evaluation_feedback(self, feedbacks):
        """Set feedback for the last behavior.

        Parameters
        ----------
        feedbacks : list of float
            feedback for each step or for the episode, depends on the problem
        """
        rewards = check_feedback(feedbacks)
        rewards = self.reward_transformation(rewards)
        q = rewards[::-1].cumsum()[::-1]
        """
        import matplotlib.pyplot as plt
        plt.plot(rewards)
        plt.plot(q)
        plt.show()
        #"""
        rollout = (q[0], self.random_state.rand(), self.params, q, self.cov)
        heapq.heappush(self.best_rollouts, rollout)

        if self.log_to_stdout or self.log_to_file:
            self.logger.info("Iteration #%d, return: %g" % (self.it, q[0]))
            self.logger.info("Variance: %g" % np.mean(self.cov))

        self.it += 1
        if self.it % self.n_samples_per_update == 0:
            self._update_variance()
            self._update_weights()

    def _update_variance(self):
        if len(self.best_rollouts) < 2:
            return

        # We use more rollouts for the variance calculation to avoid
        # rapid convergence to 0
        var_nom = np.zeros_like(self.mean)
        var_dnom = 0.0
        for _, _, params, q, _ in heapq.nlargest(30, self.best_rollouts):
            # This simplified version of the update assumes
            # * that the covariance is a diagonal matrix
            # * noise is the same over a whole rollout
            q_sum = np.sum(q)
            var_nom += q_sum * (params - self.mean) ** 2
            var_dnom += q_sum
        # TODO variance is growing unreasonably large without devision by 10, why?
        self.cov = var_nom / (10 * var_dnom + 1e-10)

    def _update_weights(self):
        n_features = self.dmp_behavior.n_features
        n_task_dims = self.dmp_behavior.n_task_dims

        best_rollouts = heapq.nlargest(
            self.n_samples_per_update, self.best_rollouts)

        param_nom = np.empty(n_features)
        param_dnom = np.empty((n_features, n_features))
        for d in range(n_task_dims):
            lo = d * n_features
            hi = (1 + d) * n_features
            param_nom[:] = 0
            param_dnom[:, :] = 0
            for _, _, params, q, cov in best_rollouts:
                cov_d = cov[lo:hi]
                W = np.array([self.tmp_outer[i] /
                              self.basis[i].T.dot(cov_d * self.basis[i])
                              for i in range(self.basis.shape[0])])
                epsilon = params[lo:hi] - self.mean[lo:hi]

                param_nom += np.sum(W.dot(epsilon) * q[:, np.newaxis], axis=0)
                param_dnom += np.sum(W * q[:, np.newaxis, np.newaxis], axis=0)

            inv_param_dnom = np.linalg.pinv(param_dnom)
            self.mean[lo:hi] += param_nom.dot(inv_param_dnom)

    def is_behavior_learning_done(self):
        """Check if the behavior learning is finished, e.g. it converged.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """
        return False

    def get_best_behavior(self):
        """Returns the best behavior found so far.

        Returns
        -------
        behavior : Behavior
            mapping from input to output
        """
        best_rollout = heapq.nlargest(1, self.best_rollouts)[0]
        self.dmp_behavior.set_params(best_rollout[2])
        self.dmp_behavior.reset()
        return self.dmp_behavior
