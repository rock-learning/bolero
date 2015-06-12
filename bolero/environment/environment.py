"""Environment interface."""
from abc import ABCMeta, abstractmethod
from ..utils import NonContextualException


class Environment(object):
    """Common interface for environments.

    An environment can execute actions, measure states and compute rewards.
    """
    __metaclass__ = ABCMeta

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
        n : int
            number of environment inputs
        """

    @abstractmethod
    def get_num_outputs(self):
        """Get number of environment outputs.

        Parameters
        ----------
        n : int
            number of environment outputs
        """

    @abstractmethod
    def get_outputs(self, values):
        """Get environment outputs, e.g. state of the environment.

        Parameters
        ----------
        values : array
            outputs for the environment, will be modified
        """

    @abstractmethod
    def set_inputs(self, values):
        """Set environment inputs, e.g. next action.

        Parameters
        ----------
        values : array,
            input of the environment
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
    def get_feedback(self, feedbacks):
        """Get the feedbacks for the last evaluation period.

        Parameters
        ----------
        feedbacks : array
            feedback values, will be overwritten

        Returns
        -------
        n_feedbacks : int
            number of values
        """

    @abstractmethod
    def is_behavior_learning_done(self):
        """Check if the behavior learning is finished.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """

    def get_maximum_feedback(self):
        """Returns the maximum feedback obtainable."""
        return 0.0


class ContextualEnvironment(Environment):
    """Common interface for (contextual) environments."""

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
        self.context = context
        return self.context

    @abstractmethod
    def get_num_context_dims(self):
        """Returns the number of context dimensions."""

    def get_maximum_feedback(self, context):
        """Returns the maximum feedback obtainable in given context."""
        return 0.0


class SetContext(Environment):
    """A contextual environment with a fixed context.

    Parameters
    ----------
    contextual_environment : ContextualEnvironment
        Environment that we want to wrap

    context : array-like, shape (n_context_dims,)
        Specific context
    """
    def __init__(self, contextual_environment, context):
        self.contextual_environment = contextual_environment
        self.context = context

    def init(self):
        """Initialize environment."""
        self.contextual_environment.init()
        self.contextual_environment.request_context(self.context)

    def reset(self):
        """Reset state of the environment."""
        self.contextual_environment.reset()

    def get_num_inputs(self):
        """Get number of environment inputs.

        Parameters
        ----------
        n : int
            number of environment inputs
        """
        return self.contextual_environment.get_num_inputs()

    def get_num_outputs(self):
        """Get number of environment outputs.

        Parameters
        ----------
        n : int
            number of environment outputs
        """
        return self.contextual_environment.get_num_outputs()

    def get_outputs(self, values):
        """Get environment outputs, e.g. state of the environment.

        Parameters
        ----------
        values : array
            outputs for the environment, will be modified
        """
        self.contextual_environment.get_outputs(values)

    def set_inputs(self, values):
        """Set environment inputs, e.g. next action.

        Parameters
        ----------
        values : array,
            input of the environment
        """
        self.contextual_environment.set_inputs(values)

    def step_action(self):
        """Take a step in the environment."""
        self.contextual_environment.step_action()

    def is_evaluation_done(self):
        """Check if the evaluation of the behavior is finished.

        Returns
        -------
        finished : bool
            Is the evaluation finished?
        """
        return self.contextual_environment.is_evaluation_done()

    def get_feedback(self, feedbacks):
        """Get the feedbacks for the last evaluation period.

        Parameters
        ----------
        feedbacks : array
            feedback values, will be overwritten

        Returns
        -------
        n_feedbacks : int
            number of values
        """
        return self.contextual_environment.get_feedback(feedbacks)

    def is_behavior_learning_done(self):
        """Check if the behavior learning is finished.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """
        return self.contextual_environment.is_behavior_learning_done()

    def get_maximum_feedback(self):
        """Returns the maximum feedback obtainable."""
        return self.contextual_environment.get_maximum_feedback(self.context)
