# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#          Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import yaml
import StringIO
import numpy as np
from .behavior import BlackBoxBehavior
import dmp


PERMITTED_DMP_METAPARAMETERS = ["x0", "g", "gd", "execution_time"]


def load_dmp_model(dmp, filename):
    """Load DMP model.

    Parameters
    ----------
    dmp : object
        DMP

    filename : string
        Name of YAML file
    """
    model = yaml.load(open(filename, "r"))
    dmp.name = model["name"]
    dmp.alpha_z = model["cs_alpha"]
    dmp.widths = np.array(model["rbf_widths"], dtype=np.float)
    dmp.centers = np.array(model["rbf_centers"], dtype=np.float)
    dmp.alpha_y = model["ts_alpha_z"]
    dmp.beta_y = model["ts_beta_z"]
    dmp.execution_time = model["ts_tau"]
    dmp.dt = model["ts_dt"]
    dmp.n_features = dmp.widths.shape[0]
    dmp.weights = np.array(model["ft_weights"], dtype=np.float
        ).reshape(dmp.n_task_dims, dmp.n_features).T.ravel()

    if dmp.execution_time != model["cs_execution_time"]:
        raise ValueError("Inconsistent execution times: %g != %g"
                         % (model["ts_tau"],
                            model["cs_execution_time"]))
    if dmp.dt != model["cs_dt"]:
        raise ValueError("Inconsistent execution times: %g != %g"
                         % (model["ts_dt"], model["cs_dt"]))


def save_dmp_model(dmp, filename):
    """Save DMP model.

    Parameters
    ----------
    dmp : object
        DMP

    filename : string
        Name of YAML file
    """
    model = {}
    model["name"] = dmp.name
    model["cs_alpha"] = dmp.alpha_z
    model["cs_execution_time"] = dmp.execution_time
    model["cs_dt"] = dmp.dt
    model["rbf_widths"] = dmp.widths.tolist()
    model["rbf_centers"] = dmp.centers.tolist()
    model["ts_alpha_z"] = dmp.alpha_y
    model["ts_beta_z"] = dmp.beta_y
    model["ts_tau"] = dmp.execution_time
    model["ts_dt"] = dmp.dt
    model["ft_weights"] = dmp.weights.reshape(
        dmp.n_features, dmp.n_task_dims).T.tolist()

    model_content = StringIO.StringIO()
    yaml.dump(model, model_content)
    with open(filename, "w") as f:
        f.write("---\n")
        f.write(model_content.getvalue())
        f.write("...\n")
    model_content.close()


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
        if configuration_file is None:
            self.execution_time = execution_time
            self.dt = dt
            self.n_features = n_features
        else:
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

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_task_dims = self.n_inputs / 3

        if hasattr(self, "configuration_file"):
            load_dmp_model(self, self.configuration_file)
        else:
            self.name = "Python DMP"
            self.alpha_z = dmp.calculate_alpha(0.01, self.execution_time, 0.0)
            self.widths = np.empty(self.n_features)
            self.centers = np.empty(self.n_features)
            dmp.initialize_rbf(self.widths, self.centers, self.execution_time,
                               0.0, 0.8, self.alpha_z)
            self.alpha_y = 25.0
            self.beta_y = self.alpha_y / 4.0
            self.weights = np.zeros((self.n_features, self.n_task_dims)).ravel()

        if not hasattr(self, "x0"):
            self.x0 = None
        if not hasattr(self, "x0d"):
            self.x0d = np.zeros(self.n_task_dims)
        if not hasattr(self, "x0dd"):
            self.x0dd = np.zeros(self.n_task_dims)
        if not hasattr(self, "g"):
            self.g = np.zeros(self.n_task_dims)
        if not hasattr(self, "gd"):
            self.gd = np.zeros(self.n_task_dims)
        if not hasattr(self, "gdd"):
            self.gdd = np.zeros(self.n_task_dims)

        self.reset()


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
        self.last_y[:] = inputs[:self.n_task_dims]
        self.last_yd[:] = inputs[self.n_task_dims:-self.n_task_dims]
        self.last_ydd[:] = inputs[-self.n_task_dims:]

        if self.x0 is None:
            self.x0 = self.last_y.copy()

    def get_outputs(self, outputs):
        """Get outputs of the last step.

        Parameters
        ----------
        outputs : array-like, shape = (3 * n_task_dims,)
            Contains positions, velocities and accelerations in that order.
            Each type is stored contiguously, i.e. for n_task_dims=2 the order
            would be: xxvvaa (x: position, v: velocity, a: acceleration).
        """
        outputs[:self.n_task_dims] = self.y[:]
        outputs[self.n_task_dims:-self.n_task_dims] = self.yd[:]
        outputs[-self.n_task_dims:] = self.ydd[:]

    def step(self):
        """Compute desired position, velocity and acceleration."""
        if self.n_task_dims == 0:
            return

        dmp.dmp_step(
            self.last_t, self.t,
            self.last_y, self.last_yd, self.last_ydd,
            self.y, self.yd, self.ydd,
            self.g, self.gd, self.gdd,
            self.x0, self.x0d, self.x0dd,
            self.execution_time, 0.0,
            self.weights,
            self.widths,
            self.centers,
            self.alpha_y, self.beta_y, self.alpha_z,
            0.001
        )

        if self.t == self.last_t:
            self.last_t = -1.0
        else:
            self.last_t = self.t
            self.t += self.dt

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
        return self.t <= self.execution_time

    def get_n_params(self):
        """Get number of weights.

        Returns
        -------
        n_params : int
            Number of DMP weights
        """
        return self.weights.size

    def get_params(self):
        """Get current weights.

        Returns
        -------
        params : array-like, shape = (n_params,)
            Current weights
        """
        return self.weights

    def set_params(self, params):
        """Set new weights.

        Parameters
        ----------
        params : array-like, shape = (n_params,)
            New weights
        """
        self.weights[:] = params

    def reset(self):
        """Reset DMP."""
        if self.x0 is None:
            self.last_y = np.zeros(self.n_task_dims)
        else:
            self.last_y = np.copy(self.x0)
        self.last_yd = np.copy(self.x0d)
        self.last_ydd = np.copy(self.x0dd)

        self.y = np.empty(self.n_task_dims)
        self.yd = np.empty(self.n_task_dims)
        self.ydd = np.empty(self.n_task_dims)

        self.last_t = 0.0
        self.t = 0.0

    def imitate(self, X, Xd=None, Xdd=None, alpha=0.0,
                allow_final_velocity=True):
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

        allow_final_velocity : bool, optional (default: True)
            Allow the final velocity to be greater than 0
        """
        if X.shape[2] > 1:
            warnings.warn("Imitation only accepts one demonstration.")
        if Xd is not None:
            warnings.warn("Xd is deprecated")
        if Xdd is not None:
            warnings.warn("Xdd is deprecated")

        X = X[:, :, 0].T.copy()
        dmp.imitate(np.arange(0, self.execution_time + self.dt, self.dt),
                    X.ravel(), self.weights, self.widths, self.centers,
                    alpha, self.alpha_y, self.beta_y, self.alpha_z,
                    allow_final_velocity)

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
        last_t = 0.0
        last_y = np.copy(self.x0)
        last_yd = np.copy(self.x0d)
        last_ydd = np.copy(self.x0dd)

        y = np.empty(self.n_task_dims)
        yd = np.empty(self.n_task_dims)
        ydd = np.empty(self.n_task_dims)

        Y = []
        Yd = []
        Ydd = []
        for t in np.arange(0, self.execution_time + self.dt, self.dt):
            dmp.dmp_step(
                last_t, t,
                last_y, last_yd, last_ydd,
                y, yd, ydd,
                self.g, self.gd, self.gdd,
                self.x0, self.x0d, self.x0dd,
                self.execution_time, 0.0,
                self.weights,
                self.widths,
                self.centers,
                self.alpha_y, self.beta_y, self.alpha_z,
                0.001
            )
            last_t = t
            last_y[:] = y
            last_yd[:] = yd
            last_ydd[:] = ydd
            Y.append(y.copy())
            Yd.append(yd.copy())
            Ydd.append(ydd.copy())

        return np.asarray(Y), np.asarray(Yd), np.asarray(Ydd)

    save = save_dmp_model

    def save_config(self, filename):
        """Save DMP configuration.

        Parameters
        ----------
        filename : string
            Name of YAML file
        """
        config = {}
        config["name"] = self.name
        config["dmp_execution_time"] = self.execution_time
        config["dmp_startPosition"] = self.x0.tolist()
        config["dmp_startVelocity"] = self.x0d.tolist()
        config["dmp_startAcceleration"] = self.x0dd.tolist()
        config["dmp_endPosition"] = self.g.tolist()
        config["dmp_endVelocity"] = self.gd.tolist()
        config["dmp_endAcceleration"] = self.gdd.tolist()

        config_content = StringIO.StringIO()
        yaml.dump(config, config_content)
        with open(filename, "w") as f:
            f.write("---\n")
            f.write(config_content.getvalue())
            f.write("...\n")
        config_content.close()

    def load_config(self, filename):
        """Load DMP configuration.

        Parameters
        ----------
        filename : string
            Name of YAML file
        """
        config = yaml.load(open(filename, "r"))
        self.execution_time = config["dmp_execution_time"]
        self.x0 = np.array(config["dmp_startPosition"], dtype=np.float)
        self.x0d = np.array(config["dmp_startVelocity"], dtype=np.float)
        self.x0dd = np.array(config["dmp_startAcceleration"], dtype=np.float)
        self.g = np.array(config["dmp_endPosition"], dtype=np.float)
        self.gd = np.array(config["dmp_endVelocity"], dtype=np.float)
        self.gdd = np.array(config["dmp_endAcceleration"], dtype=np.float)
