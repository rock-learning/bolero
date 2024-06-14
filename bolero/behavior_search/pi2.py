# Author: Alexander Fabisch <afabisch@informatik.uni-bremen.de>
from .behavior_search import BehaviorSearch, PickableMixin
from bolero.representation import DMPBehavior


class PI2(BehaviorSearch, PickableMixin):
    """Policy Improvements with Path Integrals (PI^2).

    .. note::

        This implementation of PI^2 only works for joint space DMPs!

    Parameters
    ----------
    TODO

    References
    ----------
    TODO
    """
    def __init__(self, behavior=None, initial_params=None, variance=1.0,
                 covariance=None, n_samples_per_update=10, log_to_file=False,
                 log_to_stdout=False, random_state=None):
        self.behavior = behavior
        self.initial_params = initial_params
        self.variance = variance
        self.covariance = covariance
        self.n_samples_per_update = n_samples_per_update
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
        if self.behavior is None:
            self.behavior = DMPBehavior()
        self.behavior.init(n_inputs, n_outputs)

        if self.initial_params is None:
            self.initial_params = np.zeros(n_params)
        else:
            self.initial_params = np.asarray(self.initial_params).astype(
                np.float64, copy=True)

        if self.covariance is None:
            self.covariance = np.eye(self.n_params)
        else:
            self.covariance = np.asarray(self.covariance).copy()
        if self.covariance.ndim == 1:
            self.covariance = np.diag(self.covariance)

        self.best_fitness = -np.inf
        self.best_fitness_it = self.it
        self.best_params = self.initial_params.copy()

        # TODO expose rbfActivations, phase from DMP implementation
        tau = self.behavior.execution_time
        dt = self.behavior.dt
        self.phases = np.array([
            #dmp.phase(t, self.behavior.alpha_z, tau, 0.0) # TODO
            self._phase(t, self.behavior.alpha_z, tau, 0.0)
            for t in np.arange(tau + dt, dt)])
        widths = self.behavior.widths
        centers = self.behavior.centers
        G = np.array([#dmp.rbf_activations(phase, widths, centers, True) * phase # TODO
                      self._rbf_activations(phase, widths, centers, True) * phase
                      for phase in self.phases])
        self.M = G ** 2
        self.M /= np.sum(self.M, axis=1)[:, np.newaxis]

        # TODO
        m = int(dmp.canonical_system.get_execution_time() /
                dmp.canonical_system.get_dt())
        m2 = n_steps - m
        N = np.hstack((np.linspace(m, 1, m), np.ones(m2)))[:, np.newaxis]
        # Final weighting vector takes the kernel activation into account
        W = np.tile(N, (1, n_params_per_dim))
        S = dmp.get_phases()
        W *= np.array([dmp.get_activations(s, normalized=False) for s in S])
        # ... and normalize through time
        W /= np.sum(W, axis=0)
        self.W = np.tile(W, (n_task_dims, 1, 1))

    def _phase(self, t, alpha_z, tau, _): # TODO from DMP impl.
        b = max(1.0 - alpha_z * 0.001 / tau, 1e-10);
        return b ** (t / 0.001)

    def _rbf_activations(self, phase, widths, centers, normalize): # TODO from DMP impl.
        activations = np.exp(-widths * (z - centers) ** 2)
        if normalized:
            activations /= activations.sum()
        return activations

    def get_next_behavior(self):
        """Obtain next behavior for evaluation.

        Returns
        -------
        behavior : Behavior
            mapping from input to output
        """
        raise NotImplementedError()

    def set_evaluation_feedback(self, feedbacks):
        """Set feedback for the last behavior.

        Parameters
        ----------
        feedbacks : list of float
            feedback for each step or for the episode, depends on the problem
        """
        raise NotImplementedError()

    def is_behavior_learning_done(self):
        """Check if the behavior learning is finished, e.g. it converged.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """
        raise NotImplementedError()

    def get_best_behavior(self):
        """Returns the best behavior found so far.

        Returns
        -------
        behavior : Behavior
            mapping from input to output
        """
        raise NotImplementedError()
