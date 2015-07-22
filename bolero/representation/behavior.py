"""Behavior interface."""
from abc import ABCMeta, abstractmethod
from ..base import Base


class Behavior(Base):
    """Behavior interface.

    A behavior maps input (e.g. state) to output (e.g. next state or action).
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def init(self, n_inputs, n_outputs):
        """Initialize the behavior.

        Parameters
        ----------
        n_inputs : int
            number of inputs

        n_outputs : int
            number of outputs
        """

    @abstractmethod
    def set_meta_parameters(self, keys, meta_parameters):
        """Set meta-parameters.

        Meta-parameters could be the goal, obstacles, ...

        Parameters
        ----------
        keys : list of string
            names of meta-parameters
        meta_parameters : list of lists of float values
            One list of floats for each parameter          
        """

    @abstractmethod
    def set_inputs(self, inputs):
        """Set input for the next step.

        If the input vector consists of positions and derivatives of these,
        by convention all positions and all derivatives should be stored
        contiguously.

        Parameters
        ----------
        inputs : array-like, shape = (n_inputs,)
            inputs, e.g. current state of the system
        """

    @abstractmethod
    def get_outputs(self, outputs):
        """Get outputs of the last step.

        If the output vector consists of positions and derivatives of these,
        by convention all positions and all derivatives should be stored
        contiguously.

        Parameters
        ----------
        outputs : array-like, shape = (n_outputs,)
            outputs, e.g. next action, will be updated
        """

    @abstractmethod
    def step(self):
        """Compute output for the received input.

        Uses the inputs and meta-parameters to compute the outputs.
        """

    def can_step(self):
        """Returns if step() can be called again.

        Returns
        -------
        can_step : bool
            Can we call step() again?
        """
        return True


class BehaviorTemplate(Base):
    """Behavior template interface."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_behavior(self, context):
        """Get behavior for a given context.

        Parameters
        ----------
        context : array-like, shape (n_context_dims,)
            Current context
        """


class BlackBoxBehavior(Behavior):
    """Can be optimized with black box optimizer.

    A behavior that can be optimized with a black box optimizer must be
    **exactly** defined by a **fixed** number of parameters.

    Parameters
    ----------
    n_inputs : int
        Number of input components.

    n_outputs : int
        Number of output components.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_n_params(self):
        """Get number of parameters.

        Returns
        -------
        n_params : int
            Number of parameters that will be optimized.
        """

    @abstractmethod
    def get_params(self):
        """Get current parameters.

        Returns
        -------
        params : array-like, shape = (n_params,)
            Current parameters.
        """

    @abstractmethod
    def set_params(self, params):
        """Set new parameter values.

        Parameters
        ----------
        params : array-like, shape = (n_params,)
            New parameters.
        """

    @abstractmethod
    def reset(self):
        """Reset behavior.

        This method is usually called after setting the parameters to reuse
        the current behavior and clear its internal state.
        """
