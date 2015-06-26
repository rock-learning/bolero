from .environment import Environment


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

    def get_feedback(self):
        """Get the feedbacks for the last evaluation period.

        Returns
        -------
        feedbacks : array
            Feedback values
        """
        return self.contextual_environment.get_feedback()

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
