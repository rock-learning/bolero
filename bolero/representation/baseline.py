# Author: Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import numpy as np
from .behavior import BlackBoxBehavior
from ..utils.validation import check_random_state


class DummyBehavior(BlackBoxBehavior):
    """Dummy behavior allows using environments which do not require behaviors.

    Some environments (e.g. the catapult environment) do not require behavior-
    search to learn actual behaviors but rather only to learn parameters
    (velocity and angle of a shoot in case of the catapult). This behavior
    encapsulates the parameters learned by the optimizer and returns them via
    get_outputs() to the environment whenever required. It thus connects
    environment and optimizer directly.
    """
    def __init__(self, **kwargs):
        self.params = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    def init(self, n_inputs, n_outputs):
        """Initialize the behavior.

        Parameters
        ----------
        n_inputs : int
            number of inputs

        n_outputs : int
            number of outputs
        """
        self.n_outputs = n_outputs
        self.params = np.ndarray(self.n_outputs, dtype=np.float64)
        if hasattr(self, "initial_params"):
            self.params[:] = self.initial_params
            self.initialized = True
        else:
            self.initialized = False

    def get_n_params(self):
        """Get number of parameters.

        Returns
        -------
        n_params : int
            Number of parameters that will be optimized.
        """
        return self.n_outputs

    def set_meta_parameters(self, keys, meta_parameters):
        """Set meta parameters (none defined for dummy behavior)."""
        if len(keys) > 0:
            raise NotImplementedError("DummyBehavior does not accept any meta "
                                      "parameters")

    def get_params(self):
        """Get current parameters.

        Returns
        -------
        params : array-like, shape = (n_params,)
            Current parameters.
        """
        if not self.initialized:
            raise ValueError("Initial parameters have not been set")
        return self.params

    def set_params(self, params):
        """Set new parameter values.

        Parameters
        ----------
        params : array-like, shape = (n_params,)
            New parameters.
        """
        self.params[:] = params
        self.initialized = True

    def set_inputs(self, inputs):
        """Set input for the next step.

        Parameters
        ----------
        inputs : array-like, shape = (0,)
            inputs, e.g. current state of the system
        """

    def get_outputs(self, outputs):
        """Get outputs of the last step.

        Parameters
        ----------
        outputs : array-like, shape = (n_outputs,)
            outputs, e.g. next action, will be updated
        """
        outputs[:] = self.params

    def step(self):
        """Does nothing in DummyBehavior."""

    def reset(self):
        """Reset behavior.

        Does nothing.
        """


class ConstantBehavior(BlackBoxBehavior):
    """Generates constant outputs.

    Parameters
    ----------
    outputs : array-like, shape (n_outputs,), optional (default: zeros)
        Values of constant outputs.
    """
    def __init__(self, outputs=None):
        self.outputs = outputs

    def init(self, n_inputs, n_outputs):
        """Initialize the behavior.

        Parameters
        ----------
        n_inputs : int
            number of inputs

        n_outputs : int
            number of outputs
        """
        self.n_outputs = n_outputs
        if self.outputs is None:
            self.outputs = np.zeros(self.n_outputs)

    def set_meta_parameters(self, keys, meta_parameters):
        """Set meta parameters (none defined for constant behavior)."""
        if len(keys) > 0:
            raise NotImplementedError("ConstantBehavior does not accept any "
                                      "meta parameters")

    def set_inputs(self, inputs):
        """Set input for the next step.

        Parameters
        ----------
        inputs : array-like, shape = (n_inputs,)
            inputs, e.g. current state of the system
        """

    def get_outputs(self, outputs):
        """Get outputs of the last step.

        Parameters
        ----------
        outputs : array-like, shape = (n_outputs,)
            outputs, e.g. next action, will be updated
        """
        outputs[:] = self.outputs

    def step(self):
        """Compute output for the received input.

        Use the inputs and meta-parameters to compute the outputs.
        """

    def get_n_params(self):
        """Get number of parameters.

        Returns
        -------
        n_params : int
            Number of parameters that will be optimized.
        """
        return 0

    def get_params(self):
        """Get current parameters.

        Returns
        -------
        params : array-like, shape = (n_params,)
            Current parameters.
        """
        return np.array([])

    def set_params(self, params):
        """Set new parameter values.

        Parameters
        ----------
        params : array-like, shape = (n_params,)
            New parameters.
        """
        if len(params) > 0:
            raise ValueError("Length of parameter vector must be 0")

    def reset(self):
        """Reset behavior.

        Does nothing.
        """


class RandomBehavior(BlackBoxBehavior):
    """Generates random outputs."""
    def __init__(self, random_state=None):
        self.random_state = random_state

    def init(self, n_inputs, n_outputs):
        """Initialize the behavior.

        Parameters
        ----------
        n_inputs : int
            number of inputs

        n_outputs : int
            number of outputs
        """
        self.n_outputs = n_outputs
        self.random_state = check_random_state(self.random_state)

    def set_meta_parameters(self, keys, meta_parameters):
        """Set meta parameters (none defined for random behavior)."""
        if len(keys) > 0:
            raise NotImplementedError("RandomBehavior does not accept any meta "
                                      "parameters")

    def set_inputs(self, inputs):
        """Set input for the next step.

        Parameters
        ----------
        inputs : array-like, shape = (n_inputs,)
            inputs, e.g. current state of the system
        """

    def get_outputs(self, outputs):
        """Get outputs of the last step.

        Parameters
        ----------
        outputs : array-like, shape = (n_outputs,)
            outputs, e.g. next action, will be updated
        """
        outputs[:] = self.random_state.randn(self.n_outputs)

    def step(self):
        """Compute output for the received input.

        Use the inputs and meta-parameters to compute the outputs.
        """

    def get_n_params(self):
        """Get number of parameters.

        Returns
        -------
        n_params : int
            Number of parameters that will be optimized.
        """
        return 0

    def get_params(self):
        """Get current parameters.

        Returns
        -------
        params : array-like, shape = (n_params,)
            Current parameters.
        """
        return np.array([])

    def set_params(self, params):
        """Set new parameter values.

        Parameters
        ----------
        params : array-like, shape = (n_params,)
            New parameters.
        """
        if len(params) > 0:
            raise ValueError("Length of parameter vector must be 0")

    def reset(self):
        """Reset behavior.

        Does nothing.
        """
