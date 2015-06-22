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

    Parameters
    ----------
    num_inputs : int
        number of inputs, should be 0

    num_outputs : int
        number of parameters
    """
    def __init__(self, num_inputs=0, num_outputs=-1, **kwargs):
        super(DummyBehavior, self).__init__(num_inputs, num_outputs)
        if "initial_params" in kwargs:
            self.__initialize_from(kwargs["initial_params"])
            self.params[:] = kwargs["initial_params"]
        else:
            self.params = None

    def __initialize_from(self, params):
        self.num_outputs = len(params)
        self.params = np.ndarray(self.num_outputs, dtype=np.float64)

    def get_n_params(self):
        """Get number of parameters.

        Returns
        -------
        n_params : int
            Number of parameters that will be optimized.
        """
        if self.num_outputs <= 0:
            raise ValueError("Initial parameters have not been set")
        return self.num_outputs

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
        if self.params is None:
            raise ValueError("Initial parameters have not been set")
        return self.params

    def set_params(self, params):
        """Set new parameter values.

        Parameters
        ----------
        params : array-like, shape = (n_params,)
            New parameters.
        """
        if self.params is None:
            self.__initialize_from(params)
        self.params[:] = params

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
        outputs : array-like, shape = (num_outputs,)
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
    num_inputs : int
        number of inputs

    num_outputs : int
        number of outputs

    outputs : array-like, shape (num_outputs,), optional (default: zeros)
        Values of constant outputs.
    """
    def __init__(self, num_inputs=0, num_outputs=0, outputs=None):
        super(ConstantBehavior, self).__init__(num_inputs, num_outputs)

        self.outputs = outputs
        if self.outputs is None:
            self.outputs = np.zeros(self.num_outputs)

    def set_meta_parameters(self, keys, meta_parameters):
        """Set meta parameters (none defined for constant behavior)."""
        if len(keys) > 0:
            raise NotImplementedError("ConstantBehavior does not accept any "
                                      "meta parameters")

    def set_inputs(self, inputs):
        """Set input for the next step.

        Parameters
        ----------
        inputs : array-like, shape = (num_inputs,)
            inputs, e.g. current state of the system
        """

    def get_outputs(self, outputs):
        """Get outputs of the last step.

        Parameters
        ----------
        outputs : array-like, shape = (num_outputs,)
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
    """Generates random outputs.

    Parameters
    ----------
    num_inputs : int
        number of inputs

    num_outputs : int
        number of outputs
    """
    def __init__(self, num_inputs=0, num_outputs=0, random_state=None):
        super(RandomBehavior, self).__init__(num_inputs, num_outputs)
        self.random_state = check_random_state(random_state)

    def set_meta_parameters(self, keys, meta_parameters):
        """Set meta parameters (none defined for random behavior)."""
        if len(keys) > 0:
            raise NotImplementedError("RandomBehavior does not accept any meta "
                                      "parameters")

    def set_inputs(self, inputs):
        """Set input for the next step.

        Parameters
        ----------
        inputs : array-like, shape = (num_inputs,)
            inputs, e.g. current state of the system
        """

    def get_outputs(self, outputs):
        """Get outputs of the last step.

        Parameters
        ----------
        outputs : array-like, shape = (num_outputs,)
            outputs, e.g. next action, will be updated
        """
        outputs[:] = self.random_state.randn(self.num_outputs)

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
