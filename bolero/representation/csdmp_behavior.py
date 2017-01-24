# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#          Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import yaml
import StringIO
import numpy as np
from .behavior import BlackBoxBehavior
from .dmp_behavior import load_dmp_model, save_dmp_model
import dmp


PERMITTED_CSDMP_METAPARAMETERS = ["x0", "g", "gd", "q0", "qg", "execution_time"]


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

    Note that it is possible to change the trajectory significantly by setting
    the start and goal. However, do not expect to be able to convert the DMP
    between coordinate frames by setting only the start and goal. Because
    the position and the orientation parts are handled separately, this will
    result in completely different trajectories.

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
        if n_inputs != 7:
            raise ValueError("Number of inputs must be 7")
        if n_outputs != 7:
            raise ValueError("Number of outputs must be 7")

        self.n_inputs = 7
        self.n_outputs = 7
        self.n_task_dims = 6

        if hasattr(self, "configuration_file"):
            load_dmp_model(self, self.configuration_file)
        else:
            self.name = "Python CSDMP"
            self.alpha_z = dmp.calculate_alpha(0.01, self.execution_time, 0.0)
            self.widths = np.empty(self.n_features)
            self.centers = np.empty(self.n_features)
            dmp.initialize_rbf(self.widths, self.centers, self.execution_time,
                               0.0, 0.8, self.alpha_z)
            self.alpha_y = 25.0
            self.beta_y = self.alpha_y / 4.0
            self.position_weights = np.empty((self.n_features, 3))
            self.orientation_weights = np.empty((self.n_features, 3))
            self.weights = np.zeros((self.n_features, self.n_task_dims))

        if not hasattr(self, "x0"):
            self.x0 = np.zeros(3)
        if not hasattr(self, "x0d"):
            self.x0d = np.zeros(3)
        if not hasattr(self, "x0dd"):
            self.x0dd = np.zeros(3)

        if not hasattr(self, "g"):
            self.g = np.zeros(3)
        if not hasattr(self, "gd"):
            self.gd = np.zeros(3)
        if not hasattr(self, "gdd"):
            self.gdd = np.zeros(3)

        if not hasattr(self, "q0"):
            self.q0 = np.array([0.0, 1.0, 0.0, 0.0])
        if not hasattr(self, "q0d"):
            self.q0d = np.zeros(3)
        if not hasattr(self, "q0dd"):
            self.q0dd = np.zeros(3)

        if not hasattr(self, "qg"):
            self.qg = np.array([0.0, 1.0, 0.0, 0.0])
        if not hasattr(self, "qgd"):
            self.qgd = np.zeros(3)
        if not hasattr(self, "qgdd"):
            self.qgdd = np.zeros(3)

        self.reset()

    def get_weights(self):
        return np.hstack((self.position_weights, self.orientation_weights))

    def set_weights(self, weights):
        if not hasattr(self, "position_weights"):
            self.position_weights = np.empty((self.n_features, 3))
        if not hasattr(self, "orientation_weights"):
            self.orientation_weights = np.empty((self.n_features, 3))

        self.position_weights[:] = weights[:, :3]
        self.orientation_weights[:] = weights[:, 3:]

    weights = property(get_weights, set_weights)

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

    def set_inputs(self, inputs):
        """Set input for the next step.

        Parameters
        ----------
        inputs : array-like, shape = (7,)
            Contains positions and rotations represented by quaternions,
            order (order: x, y, z, w, rx, ry, rz)
        """
        self.last_y[:] = inputs[:3]
        self.last_r[:] = inputs[3:]
        self.last_yd[:] = self.yd
        self.last_rd[:] = self.rd
        self.last_ydd[:] = self.ydd
        self.last_rdd[:] = self.rdd

    def get_outputs(self, outputs):
        """Get outputs of the last step.

        Parameters
        ----------
        outputs : array-like, shape = (7,)
            Contains positions and rotations represented by quaternions,
            order (order: x, y, z, w, rx, ry, rz)
        """
        outputs[:3] = self.y
        outputs[3:] = self.r

    def step(self):
        """Compute desired position, velocity and acceleration."""
        dmp.dmp_step(
            self.last_t, self.t,
            self.last_y, self.last_yd, self.last_ydd,
            self.y, self.yd, self.ydd,
            self.g, self.gd, self.gdd,
            self.x0, self.x0d, self.x0dd,
            self.execution_time, 0.0,
            self.position_weights,
            self.widths,
            self.centers,
            self.alpha_y, self.beta_y, self.alpha_z,
            0.001
        )

        dmp.quaternion_dmp_step(
            self.last_t, self.t,
            self.last_r, self.last_rd, self.last_rdd,
            self.r, self.rd, self.rdd,
            self.qg, self.qgd, self.qgdd,
            self.q0, self.q0d, self.q0dd,
            self.execution_time, 0.0,
            self.orientation_weights,
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
        return self.weights.ravel()

    def set_params(self, params):
        """Set new weights.

        Parameters
        ----------
        params : array-like, shape = (n_params,)
            New weights
        """
        self.weights = params.reshape(self.n_features, 6)

    def reset(self):
        """Reset DMP."""
        self.last_y = np.copy(self.x0)
        self.last_yd = np.copy(self.x0d)
        self.last_ydd = np.copy(self.x0dd)

        self.y = np.empty(3)
        self.yd = np.empty(3)
        self.ydd = np.empty(3)

        self.last_r = np.copy(self.q0)
        self.last_rd = np.copy(self.q0d)
        self.last_rdd = np.copy(self.q0dd)

        self.r = np.empty(4)
        self.rd = np.empty(3)
        self.rdd = np.empty(3)

        self.last_t = 0.0
        self.t = 0.0

    def imitate(self, X, alpha=0.0, allow_final_velocity=True):
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

        allow_final_velocity : bool, optional (default: True)
            Allow the final velocity to be greater than 0
        """
        X_pos = X[:3, :].T.copy()
        X_rot = X[3:, :].T.copy()
        dmp.imitate(
            np.arange(0, self.execution_time + self.dt, self.dt),
            X_pos, self.position_weights, self.widths, self.centers,
            alpha, self.alpha_y, self.beta_y, self.alpha_z,
            allow_final_velocity)
        dmp.quaternion_imitate(
            np.arange(0, self.execution_time + self.dt, self.dt),
            X_rot, self.orientation_weights, self.widths, self.centers,
            alpha, self.alpha_y, self.beta_y, self.alpha_z,
            allow_final_velocity)

    def trajectory(self):
        """Generate trajectory represented by the DMP in open loop.

        The function can be used for debugging purposes.

        Returns
        -------
        X : array, shape (n_steps, 7)
            Positions and rotations (order: x, y, z, w, rx, ry, rz)
        """
        last_t = 0.0

        last_y = np.copy(self.x0)
        last_yd = np.copy(self.x0d)
        last_ydd = np.copy(self.x0dd)

        y = np.empty(3)
        yd = np.empty(3)
        ydd = np.empty(3)

        last_r = np.copy(self.q0)
        last_rd = np.copy(self.q0d)
        last_rdd = np.copy(self.q0dd)

        r = np.empty(4)
        rd = np.empty(3)
        rdd = np.empty(3)

        Y = []
        R = []
        for t in np.arange(0, self.execution_time + self.dt, self.dt):
            dmp.dmp_step(
                last_t, t,
                last_y, last_yd, last_ydd,
                y, yd, ydd,
                self.g, self.gd, self.gdd,
                self.x0, self.x0d, self.x0dd,
                self.execution_time, 0.0,
                self.position_weights,
                self.widths,
                self.centers,
                self.alpha_y, self.beta_y, self.alpha_z,
                0.001
            )

            dmp.quaternion_dmp_step(
                last_t, t,
                last_r, last_rd, last_rdd,
                r, rd, rdd,
                self.qg, self.qgd, self.qgdd,
                self.q0, self.q0d, self.q0dd,
                self.execution_time, 0.0,
                self.orientation_weights,
                self.widths,
                self.centers,
                self.alpha_y, self.beta_y, self.alpha_z,
                0.001
            )

            last_t = t
            last_y[:] = y
            last_yd[:] = yd
            last_ydd[:] = ydd
            last_r[:] = r
            last_rd[:] = rd
            last_rdd[:] = rdd
            Y.append(y.copy())
            R.append(r.copy())

        return np.hstack((Y, R))

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
        config["executionTime"] = self.execution_time
        config["startPosition"] = self.x0.tolist()
        config["startVelocity"] = self.x0d.tolist()
        config["startAcceleration"] = self.x0dd.tolist()
        config["startRotation"] = self.q0.tolist()
        config["startAngularVelocity"] = self.q0d.tolist()
        config["endPosition"] = self.g.tolist()
        config["endVelocity"] = self.gd.tolist()
        config["endAcceleration"] = self.gdd.tolist()
        config["endRotation"] = self.qg.tolist()

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
        self.execution_time = config["executionTime"]
        self.x0 = np.array(config["startPosition"], dtype=np.float)
        self.x0d = np.array(config["startVelocity"], dtype=np.float)
        self.x0dd = np.array(config["startAcceleration"], dtype=np.float)
        self.q0 = np.array(config["startRotation"], dtype=np.float)
        self.q0d = np.array(config["startAngularVelocity"], dtype=np.float)
        self.g = np.array(config["endPosition"], dtype=np.float)
        self.gd = np.array(config["endVelocity"], dtype=np.float)
        self.gdd = np.array(config["endAcceleration"], dtype=np.float)
        self.qg = np.array(config["endRotation"], dtype=np.float)
