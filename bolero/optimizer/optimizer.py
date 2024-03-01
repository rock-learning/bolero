"""Optimizer interface."""
from abc import ABCMeta, abstractmethod
from ..utils import NonContextualException
from ..base import Base


ABC = ABCMeta('ABC', (object,), {'__slots__': ()})


class ContextualOptimizer(Base, ABC):
    """Common interface for (contextual) optimizers.

    This is a simple derivative-free parameter optimizer.
    """

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
    def get_next_parameters(self, params):
        """Get next individual/parameter vector for evaluation.

        Parameters
        ----------
        params : array_like, shape (n_params,)
            Parameter vector, will be modified
        """

    @abstractmethod
    def set_evaluation_feedback(self, rewards):
        """Set feedbacks for the parameter vector.

        Parameters
        ----------
        rewards : list of float
            Feedbacks for each step or for the episode, depends on the problem
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
        context : array-like, shape (n_context_dims,)
            The context in which the next rollout will be performed
        """

    @abstractmethod
    def best_policy(self):
        """Return current best estimate of contextual policy.

        Returns
        -------
        policy : UpperLevelPolicy
            Best estimate of upper-level policy
        """


class Optimizer(Base, ABC):
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
    def get_next_parameters(self, params):
        """Get next individual/parameter vector for evaluation.

        Parameters
        ----------
        params : array_like, shape (n_params,)
            Parameter vector, will be modified
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
        """Get best individual/parameter vector so far.

        Returns
        -------
        p : array_like, shape (n_params,)
            Best parameter vector so far
        """

    @abstractmethod
    def get_next_parameter_batch(self, params, dimension, batch_size):
        """Get a batch of parameter sets for parallel computing.

        Returns
        -------
        p : array_like, shape (n_params*batch_size,)
        """

    @abstractmethod
    def set_batch_feedback(self, rewards, num_rewards_per_entry, batch_size):
        """Set the rewards for the last returned batch of parameter sets.

        Parameters
        ----------
        rewards : list of float
            feedbacks for each parameter set evaluated for one episode. The
            length of the list is batch_size*num_rewards_per_batch
        num_rewards_per_batch: the number of rewards for individual parameter
            set
        """

    @abstractmethod
    def get_batch_size(self):
        """Set the rewards for the last returned batch of parameter sets.

        Parameters
        ----------
        batch_size : number of parameter sets per batch
        """
