"""Optimizer interface."""
from abc import ABCMeta, abstractmethod
from ..utils import NonContextualException


class ContextualOptimizer(object):
    """Common interface for (contextual) optimizers.

    This is a simple derivative-free parameter optimizer.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def init(self, n_params, n_context_dims):
        """Initialize optimizer.

        Parameters
        ----------
        n_params : int
            dimension of the parameter vector

        n_context_dims : int
            number of dimensions of the context space
        """

    @abstractmethod
    def get_next_parameters(self, p):
        """Get next individual/parameter vector for evaluation.

        Parameters
        ----------
        p : array_like, shape (num_p,)
            parameter vector, will be modified
        """

    @abstractmethod
    def set_evaluation_feedback(self, rewards):
        """Set feedbacks for the parameter vector.

        Parameters
        ----------
        rewards : list of float
            feedbacks for each step or for the episode, depends on the problem
        """

    @abstractmethod
    def is_behavior_learning_done(self):
        """Check if the optimization is finished.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """

    @abstractmethod
    def get_desired_context(self):
        """Chooses desired context for next evaluation.

        Returns
        -------
        context : ndarray-like, default=None
            The context in which the next rollout shall be performed. If None,
            the environment may select the next context without any preferences.
        """

    @abstractmethod
    def set_context(self, context):
        """Set context of next evaluation.

        Note that the set context need not necessarily be the same that was
        requested by get_desired_context().

        Parameters
        ----------
        context : ndarray-like
            The context in which the next rollout will be performed
        """

    @abstractmethod
    def best_policy(self):
        """Return current best estimate of contextual policy. """


class Optimizer(object):
    """Common interface for (non-contextual) optimizers."""

    @abstractmethod
    def init(self, n_params):
        """Initialize the behavior search.

        Parameters
        ----------
        n_params : int
            dimension of the parameter vector
        """

    @abstractmethod
    def get_next_parameters(self, p):
        """Get next individual/parameter vector for evaluation.

        Parameters
        ----------
        p : array_like, shape (num_p,)
            parameter vector, will be modified
        """

    @abstractmethod
    def set_evaluation_feedback(self, rewards):
        """Set feedbacks for the parameter vector.

        Parameters
        ----------
        rewards : list of float
            feedbacks for each step or for the episode, depends on the problem
        """

    @abstractmethod
    def is_behavior_learning_done(self):
        """Check if the optimization is finished.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """

    @abstractmethod
    def get_best_parameters(self):
        """Method not supported by ContextualOptimizer.

        For contextual optimizers, this method cannot meaningfully implemented
        since the best parameters depend on the context. Instead, a method
        best_policy() needs to be implemented which returns the best policy,
        where a policy implements a mapping from context onto parameters.
        """
