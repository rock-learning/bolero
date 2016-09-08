# Author: Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import numpy as np
from .behavior import BlackBoxBehavior


class LinearBehavior(BlackBoxBehavior):
    """Linear mapping from inputs to outputs.

    Parameters
    ----------
    n_inputs : int
        Number of input components.

    n_outputs : int
        Number of output components.
    """
    def init(self, n_inputs, n_outputs):
        """Initialize the behavior.

        Parameters
        ----------
        n_inputs : int
            number of inputs

        n_outputs : int
            number of outputs
        """
        self.inputs = np.empty(n_inputs)
        self.inputs[:] = np.nan
        self.outputs = np.empty(n_outputs)
        self.outputs[:] = np.nan
        self.W = np.zeros((n_outputs, n_inputs + 1))

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

    def set_inputs(self, inputs):
        """Set input for the next step.

        Parameters
        ----------
        inputs : array-like, shape = (n_inputs,)
            inputs, e.g. current state of the system
        """
        self.inputs[:] = inputs

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
        outputs[:] = self.outputs

    def step(self):
        """Compute output for the received input.

        Uses the inputs and meta-parameters to compute the outputs.
        """
        inputs = np.hstack((self.inputs, (1,)))
        self.outputs[:] = self.W.dot(inputs)

    def can_step(self):
        """Returns if step() can be called again.

        Returns
        -------
        can_step : bool
            Can we call step() again?
        """
        return True

    def get_n_params(self):
        """Get number of parameters.

        Returns
        -------
        n_params : int
            Number of parameters that will be optimized.
        """
        return self.W.size

    def get_params(self):
        """Get current parameters.

        Returns
        -------
        params : array-like, shape = (n_params,)
            Current parameters.
        """
        return self.W.ravel()

    def set_params(self, params):
        """Set new parameter values.

        Parameters
        ----------
        params : array-like, shape = (n_params,)
            New parameters.
        """
        self.W[:, :] = params.reshape(self.W.shape)

    def reset(self):
        """Reset behavior.

        This method is usually called after setting the parameters to reuse
        the current behavior and clear its internal state.
        """
