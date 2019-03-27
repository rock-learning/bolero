"""Environment interface."""
from abc import ABCMeta, abstractmethod
from ..utils import NonContextualException
from ..base import Base


ABC = ABCMeta('ABC', (object,), {'__slots__': ()})


class Environment(Base, ABC):
    """Common interface for environments.

    An environment can execute actions, measure states and compute rewards.
    It defines a learning problem.
    """

    @abstractmethod
    def init(self):
        """Initialize environment."""

    @abstractmethod
    def reset(self):
        """Reset state of the environment."""

    @abstractmethod
    def get_num_inputs(self):
        """Get number of environment inputs.

        Parameters
        ----------
        n_inputs : int
            Number of environment inputs
        """

    @abstractmethod
    def get_num_outputs(self):
        """Get number of environment outputs.

        Parameters
        ----------
        n_outputs : int
            Number of environment outputs
        """

    @abstractmethod
    def get_outputs(self, values):
        """Get environment outputs, e.g. state of the environment.

        Parameters
        ----------
        values : array
            Outputs of the environment, will be modified
        """

    @abstractmethod
    def set_inputs(self, values):
        """Set environment inputs, e.g. next action.

        Parameters
        ----------
        values : array,
            Input of the environment
        """

    @abstractmethod
    def step_action(self):
        """Take a step in the environment."""

    @abstractmethod
    def is_evaluation_done(self):
        """Check if the evaluation of the behavior is finished.

        Returns
        -------
        finished : bool
            Is the evaluation finished?
        """

    @abstractmethod
    def get_feedback(self):
        """Get the feedbacks for the last evaluation period.

        Returns
        -------
        feedbacks : array
            Feedback values
        """

    @abstractmethod
    def is_behavior_learning_done(self):
        """Check if the behavior learning is finished.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """

    @abstractmethod
    def get_maximum_feedback(self):
        """Returns the maximum sum of feedbacks obtainable."""


class ContextualEnvironment(Environment):
    """Common interface for (contextual) environments."""

    @abstractmethod
    def request_context(self, context):
        """Request that a specific context is used.

        Parameters
        ----------
        context : array-like, shape (n_context_dims,)
            The requested context that shall be used in the next rollout.
            Defaults to None. In that case, the environment selects the next
            context

        Returns
        -------
        context : array-like, shape (n_context_dims,)
            The actual context used in the next rollout. This may or
            may not be the requested context, depending on the respective
            environment.
        """

    @abstractmethod
    def get_num_context_dims(self):
        """Returns the number of context dimensions."""

    @abstractmethod
    def get_maximum_feedback(self, context):
        """Returns the maximum sum of feedbacks obtainable in given context."""
