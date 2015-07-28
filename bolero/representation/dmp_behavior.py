# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#          Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import numpy as np
from .behavior import BlackBoxBehavior
from dmp import DMP, RbDMP, imitate_dmp


PERMITTED_DMP_METAPARAMETERS = ["x0", "g", "gd", "execution_time"]
PERMITTED_CSDMP_METAPARAMETERS = ["x0", "g", "gd", "q0", "qg", "execution_time"]


class DMPBehavior(BlackBoxBehavior):
    """Dynamical Movement Primitive.

    Can be used to optimize the weights of a DMP with a black box optimizer.
    This is a wrapper for the optional DMP module of bolero. Only the weights
    of the DMP will be optimized. To optimize meta-parameters like the goal or
    the goal velocity, you have to implement your own wrapper. This can be a
    subclass of this wrapper that only overrides the methods that provide
    access to the parameters.

    An object can be created either by passing a configuration file or the
    specification of a DMP. A DMP configuration file describes all parameters
    of the DMP model and it is not recommended to generate it manually.

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
        self.execution_time = execution_time
        self.dt = dt
        self.n_features = n_features
        self.configuration_file = configuration_file

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

        if self.configuration_file is None:
            self.dmp = DMP(execution_time=self.execution_time, dt=self.dt,
                           n_features=self.n_features)
        else:
            self.dmp = DMP.from_file(self.configuration_file)

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
            if key not in PERMITTED_DMP_METAPARAMETERS:
                raise ValueError(
                    "Meta parameter '%s' is not allowed, use one of %r"
                    % (key, PERMITTED_DMP_METAPARAMETERS))
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

    def imitate(self, X, Xd=None, Xdd=None, alpha=0.0):
        """Learn weights of the DMP from demonstrations.

        Parameters
        ----------
        X : array, shape (n_task_dims, n_steps, n_demos)
            The demonstrated trajectories to be imitated.

        Xd : array, shape (n_task_dims, n_steps, n_demos), optional
            Velocities of the demonstrated trajectories.

        Xdd : array, shape (n_task_dims, n_steps, n_demos), optional
            Accelerations of the demonstrated trajectories.

        alpha : float >= 0, optional (default: 0)
            The ridge parameter of linear regression. Small positive values of
            alpha improve the conditioning of the problem and reduce the
            variance of the estimates.
        """
        imitate_dmp(self.dmp, X, Xd, Xdd, alpha=alpha, set_weights=True)

    def trajectory(self):
        """Generate trajectory represented by the DMP in open loop.

        The function can be used for debugging purposes.

        Returns
        -------
        X : array, shape (n_steps, n_task_dims)
            Positions

        Xd : array, shape (n_steps, n_task_dims)
            Velocities

        Xdd : array, shape (n_steps, n_task_dims)
            Accelerations
        """
        x, xd, xdd = (np.copy(self.x0), np.zeros_like(self.x0),
                      np.zeros_like(self.x0))
        X, Xd, Xdd = [], [], []

        self.dmp.reset()
        while self.dmp.can_step():
            x, xd, xdd = self.dmp.execute_step(x, xd, xdd, self.x0, self.g,
                                               self.gd)
            X.append(x.copy())
            Xd.append(xd.copy())
            Xdd.append(xdd.copy())

        return np.array(X), np.array(Xd), np.array(Xdd)


class CartesianDMPBehavior(BlackBoxBehavior):
    """Cartesian Space Dynamical Movement Primitive.

    Can be used to optimize the weights of a Cartesian Space DMP with a black
    box optimizer. This is a wrapper for the optional DMP module of bolero.
    Only the weights of the Cartesian Space DMP will be optimized. To optimize
    meta-parameters like the goal or the goal velocity, you have to implement
    your own wrapper. This can be a subclass of this wrapper that only
    overrides the methods that provide access to the parameters.

    An object can be created either by passing a configuration file or the
    specification of a Cartesian Space DMP. A Cartesian Space DMP configuration
    file describes all parameters of the DMP model and it is not recommended to
    generate it manually.

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
        self.execution_time = execution_time
        self.dt = dt
        self.n_features = n_features
        self.configuration_file = configuration_file

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

        if self.configuration_file is None:
            self.dmp = RbDMP(execution_time=self.execution_time, dt=self.dt,
                             n_features=self.n_features)
        else:
            self.dmp = RbDMP.from_file(self.configuration_file)
            # TODO implement

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.x0, self.x0d, self.x0dd = np.zeros(3), np.zeros(3), np.zeros(3)
        self.g, self.gd, self.gdd = np.zeros(3), np.zeros(3), np.zeros(3)
        self.q0, self.q0d = (np.array([0.0, 1.0, 0.0, 0.0]),
                             np.array([0.0, 0.0, 0.0]))
        self.qg = np.array([0.0, 1.0, 0.0, 0.0])

        self.x = np.empty(7)
        self.v = np.zeros(3)
        self.a = np.zeros(3)

        self.dmp.configure(self.x0, self.x0d, self.x0dd, self.q0, self.q0d,
                           self.g, self.gd, self.gdd, self.qg,
                           self.execution_time)

    def set_meta_parameters(self, keys, meta_parameters):
        """Set DMP meta parameters.

        Permitted meta-parameters:

        x0 : array
            Initial position

        g : array
            Goal

        gd : array
            Velocity at the goal

        q0 : array
            Initial rotation

        qg : array
            Final rotation

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
            if key not in PERMITTED_CSDMP_METAPARAMETERS:
                raise ValueError(
                    "Meta parameter '%s' is not allowed, use one of %r"
                    % (key, PERMITTED_CSDMP_METAPARAMETERS))
            setattr(self, key, meta_parameter)
        self.dmp.configure(self.x0, self.x0d, self.x0dd, self.q0, self.q0d,
                           self.g, self.gd, self.gdd, self.qg,
                           self.execution_time)

    def set_inputs(self, inputs):
        """Set input for the next step.

        Parameters
        ----------
        inputs : array-like, shape = (7,)
            Contains positions and rotations represented by quaternions,
            order (order: x, y, z, w, rx, ry, rz)
        """
        self.x[:] = inputs[:]

    def get_outputs(self, outputs):
        """Get outputs of the last step.

        Parameters
        ----------
        outputs : array-like, shape = (7,)
            Contains positions and rotations represented by quaternions,
            order (order: x, y, z, w, rx, ry, rz)
        """
        outputs[:] = self.x[:]

    def step(self):
        """Compute desired position, velocity and acceleration."""
        if self.dmp.can_step():
            self.x[:3], self.v, self.a, self.x[3:] = self.dmp.execute_step(
                self.x[:3], self.v, self.a, self.x[3:])

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

    def imitate(self, X, alpha=0.0):
        """Learn weights of the DMP from demonstrations.

        Parameters
        ----------
        X : array, shape (7, n_steps)
            The demonstrated trajectory (order: x, y, z, w, rx, ry, rz) to be
            imitated.

        alpha : float >= 0, optional (default: 0)
            The ridge parameter of linear regression. Small positive values of
            alpha improve the conditioning of the problem and reduce the
            variance of the estimates.
        """
        imitate_dmp(self.dmp, X, alpha=alpha, set_weights=True)

    def trajectory(self):
        """Generate trajectory represented by the DMP in open loop.

        The function can be used for debugging purposes.

        Returns
        -------
        X : array, shape (n_steps, 7)
            Positions and rotations (order: x, y, z, w, rx, ry, rz)
        """
        x, xd, xdd, q = (np.copy(self.x0), np.zeros(3), np.zeros(3),
                         np.copy(self.q0))
        X, Q = [], []

        self.dmp.reset()
        while self.dmp.can_step():
            x, xd, xdd, q = self.dmp.execute_step(x, xd, xdd, q)
            X.append(x.copy())
            Q.append(q.copy())

        return np.hstack((X, Q))
