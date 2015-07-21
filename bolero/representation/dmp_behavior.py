# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#          Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import numpy as np
from .behavior import BlackBoxBehavior
from dmp import DMP


PERMITTED_METAPARAMETERS = ["x0", "g", "gd", "execution_time"]


class DMPBehavior(BlackBoxBehavior):
    """Dynamical Movement Primitive.

    Can be used to optimize the weights of a DMP with a black box optimizer.
    This is a wrapper for the optional DMP module of bolero. Only the weights
    of the DMP will be optimized. To optimize meta-parameters like the goal or
    the goal velocity, you have to implement your own wrapper. This can be a
    subclass of this wrapper that only overrides the methods that provide
    access to the parameters.

    An object can be created either by passing a configuration file or a DMP
    object. A DMP configuration file describes all parameters of the DMP model
    and it is not recommended to generate it manually.

    Parameters
    ----------
    execution_time : float, optional (default: 1)
        Execution time of the DMP in seconds.

    dt : float, optional (default: 0.01)
        Time between successive steps in seconds.

    n_features : int, optional (default: 50)
        Number of RBF features for each dimension of the DMP.

    configuration_file : string, optional (default: None)
        Name of a configuration file that should be used to initialize the DMP.
        If it is set all other arguments will be ignored.
    """
    def __init__(self, execution_time=1.0, dt=0.01, n_features=50,
                 configuration_file=None):
        if configuration_file is None:
            self.dmp = DMP(execution_time=execution_time, dt=dt,
                           n_features=n_features)
        else:
            self.dmp = DMP.from_file(configuration_file)

    def init(self, n_inputs, n_outputs):
        """Initialize the behavior.

        Parameters
        ----------
        n_inputs : int
            number of inputs

        n_outputs : int
            number of outputs
        """
        if n_inputs != n_outputs:
            raise ValueError("Input and output dimensions must match, got "
                             "%d inputs and %d outputs" % (n_inputs, n_outputs))

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.n_task_dims = self.n_inputs / 3

        if self.dmp.get_weights() is None:
            weights = np.zeros((self.n_task_dims, self.dmp.get_num_features()))
            self.dmp.set_weights(weights)

        self.x0 = None
        self.g = np.zeros(self.n_task_dims)
        self.gd = None

        self.x = np.empty(self.n_task_dims)
        self.v = np.empty(self.n_task_dims)
        self.a = np.empty(self.n_task_dims)

    def set_meta_parameters(self, keys, meta_parameters):
        """Set DMP meta parameters.

        Permitted meta-parameters:

        x0 : array
            Initial position

        g : array
            Goal

        gd : array
            Velocity at the goal

        execution_time : float
            New execution time

        Parameters
        ----------
        keys : list of string
            names of meta-parameters

        meta_parameters : list of float
            values of meta-parameters
        """
        for key, meta_parameter in zip(keys, meta_parameters):
            if key not in PERMITTED_METAPARAMETERS:
                raise ValueError("Meta parameter '%s' is not allowed, use "
                                 "one of %r" % (key, PERMITTED_METAPARAMETERS))
            setattr(self, key, meta_parameter)
        self.dmp.set_metaparameters(keys, meta_parameters)

    def set_inputs(self, inputs):
        """Set input for the next step.

        In case the start position (x0) has not been set as a meta-parameter
        we take the first position as x0.

        Parameters
        ----------
        inputs : array-like, shape = (3 * n_task_dims,)
            Contains positions, velocities and accelerations in that order.
            Each type is stored contiguously, i.e. for n_task_dims=2 the order
            would be: xxvvaa (x: position, v: velocity, a: acceleration).
        """
        n_task_dims = len(inputs) / 3
        self.x[:] = inputs[:n_task_dims]
        self.v[:] = inputs[n_task_dims:-n_task_dims]
        self.a[:] = inputs[-n_task_dims:]

        if self.x0 is None:
            self.x0 = self.x.copy()

    def get_outputs(self, outputs):
        """Get outputs of the last step.

        Parameters
        ----------
        outputs : array-like, shape = (3 * n_task_dims,)
            Contains positions, velocities and accelerations in that order.
            Each type is stored contiguously, i.e. for n_task_dims=2 the order
            would be: xxvvaa (x: position, v: velocity, a: acceleration).
        """
        n_task_dims = len(outputs) / 3
        outputs[:n_task_dims] = self.x[:]
        outputs[n_task_dims:-n_task_dims] = self.v[:]
        outputs[-n_task_dims:] = self.a[:]

    def step(self):
        """Compute desired position, velocity and acceleration."""
        if self.n_task_dims == 0:
            return

        if self.dmp.can_step():
            self.x, self.v, self.a = self.dmp.execute_step(
                self.x, self.v, self.a, self.x0, self.g, self.gd)
        else:
            self.v[:] = 0.0
            self.a[:] = 0.0

    def can_step(self):
        """Returns if step() can be called again.

        Note that calling step() after this function returns False will not
        result in an error. The velocity and acceleration will be set to 0
        and we hold the last position instead.

        Returns
        -------
        can_step : bool
            Can we call step() again?
        """
        return self.dmp.can_step()

    def get_n_params(self):
        """Get number of weights.

        Returns
        -------
        n_params : int
            Number of DMP weights
        """
        return self.dmp.get_weights().size

    def get_params(self):
        """Get current weights.

        Returns
        -------
        params : array-like, shape = (n_params,)
            Current weights
        """
        return self.dmp.get_weights().ravel()

    def set_params(self, params):
        """Set new weights.

        Parameters
        ----------
        params : array-like, shape = (n_params,)
            New weights
        """
        self.dmp.set_weights(params)

    def reset(self):
        """Reset DMP."""
        self.dmp.reset()
        self.x0 = None
