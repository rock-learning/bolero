# Authors: Bernd Poppinga <bernd.poppinga@dfki.de>
#          Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import yaml
import warnings
try:  # Python 2
    from StringIO import StringIO
except:  # Python 3
    from io import StringIO
import numpy as np
from .behavior import BlackBoxBehavior
import promp

PERMITTED_PROMP_METAPARAMETERS = ["x0", "g", "gd", "execution_time"]


def load_promp_model(promp, filename):
    """Load promp model.

    Parameters
    ----------
    promp : object
        promp

    filename : string
        Name of YAML file
    """
    model = yaml.load(open(filename, "r"))
    promp.name = model["name"]
    promp.data = model["data"]

    promp.dt = model["ts_dt"]
    promp.overlap = model["overlap"]
    promp.n_features = promp.widths.shape[0]
    promp.weights = np.array(
        model["ft_weights"], dtype=np.float).reshape(promp.n_task_dims,
                                                     promp.n_features).T

    if promp.execution_time != model["cs_execution_time"]:
        raise ValueError("Inconsistent execution times: %g != %g" %
                         (model["ts_tau"], model["cs_execution_time"]))
    if promp.dt != model["cs_dt"]:
        raise ValueError("Inconsistent execution times: %g != %g" %
                         (model["ts_dt"], model["cs_dt"]))


def save_promp_model(promp, filename):
    """Save promp model.

    Parameters
    ----------
    promp : object
        promp

    filename : string
        Name of YAML file
    """
    model = {}
    model["name"] = promp.name
    model["data"] = promp.data
    model["ts_tau"] = promp.execution_time
    model["ts_dt"] = promp.dt

    model_content = StringIO()
    yaml.dump(model, model_content)
    with open(filename, "w") as f:
        f.write("---\n")
        f.write(model_content.getvalue())
        f.write("...\n")
    model_content.close()


class ProMPBehavior(BlackBoxBehavior):
    """Probabilistic Movement Primitive.

    Can be used to optimize the weights of a promp with a black box optimizer.
    This is a wrapper for the optional promp module of bolero. Only the weights
    of the promp will be optimized. To optimize meta-parameters like the goal
    or the goal velocity, you have to implement your own wrapper. This can be a
    subclass of this wrapper that only overrides the methods that provide
    access to the parameters.

    An object can be created either by passing a configuration file or the
    specification of a promp. A promp configuration file describes all
    parameters of the promp model and it is not recommended to generate it
    manually.

    Parameters
    ----------
    execution_time : float, optional (default: 1)
        Execution time of the promp in seconds.

    dt : float, optional (default: 0.01)
        Time between successive steps in seconds.

    n_features : int, optional (default: 50)
        Number of RBF features for each dimension of the promp.

    configuration_file : string, optional (default: None)
        Name of a configuration file that should be used to initialize the
        promp. If it is set all other arguments will be ignored.
    n_features: int, optional (default: 50)
        How many features are to be used
    overlap: float, optional (default: 0.7)
        How much shall the gaussians which are used for approximation overlap
    """

    def __init__(self,
                 execution_time=1.0,
                 dt=0.01,
                 n_features=50,
                 overlap=0.7,
                 configuration_file=None,
                 learn_covariance=False,
                 use_covar=False):
        self.learn_covariance = learn_covariance
        self.use_covar = use_covar
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

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_task_dims = self.n_inputs / 2
        self.overlap = 0.7
        self.valueMeans = np.empty(self.n_task_dims * 2)
        self.valueCovs = np.empty((self.n_task_dims * 2)**2)
        if hasattr(self, "configuration_file"):
            load_promp_model(self, self.configuration_file)
        else:
            self.name = "Python promp"
            self.data = promp.TrajectoryData(self.n_features, self.n_task_dims,
                                             True, self.overlap)
            self.random_state_ = np.random.randint(
                100000000)  # some big number for random state

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

        self._params_set = False

        self.reset()

    def set_meta_parameters(self, keys, meta_parameters):
        """Set promp meta parameters.

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
        conditionPoints = []
        for key, meta_parameter in zip(keys, meta_parameters):
            if key not in PERMITTED_PROMP_METAPARAMETERS:
                raise ValueError(
                    "Meta parameter '%s' is not allowed, use one of %r" %
                    (key, PERMITTED_PROMP_METAPARAMETERS))
            setattr(self, key, meta_parameter)

            if key == "x0":
                for i in range(len(meta_parameter)):
                    conditionPoints += [[0, i, 0, meta_parameter[i], 0.0001]]

            elif key == "g":
                for i in range(len(meta_parameter)):
                    conditionPoints += [[1, i, 0, meta_parameter[i], 0.0001]]

            elif key == "gd":
                for i in range(len(meta_parameter)):
                    conditionPoints += [[1, i, 1, meta_parameter[i], 0.0001]]

        if len(conditionPoints) >= 1:
            conditionPoints_ = np.array(conditionPoints).flatten()
            self.data.condition(len(conditionPoints), conditionPoints_)

    def set_inputs(self, inputs):
        """Set input for the next step.

        In case the start position (x0) has not been set as a meta-parameter
        we take the first position as x0.

        Parameters
        ----------
        inputs : array-like, shape = (2 * n_task_dims,)
            Contains positions and velocities in that order.
            Each type is stored contiguously, i.e. for n_task_dims=2 the order
            would be: xxvv (x: position, v: velocity).
        """
        # just open loop by now
        # TODO change that

    def get_outputs(self, outputs):
        """Get outputs of the last step.

        Parameters
        ----------
        outputs : array-like, shape = (2 * n_task_dims,)
            Contains positions, velocities in that order.
            Each type is stored contiguously, i.e. for n_task_dims=2 the order
            would be: xxvv (x: position, v: velocity).
            If the covariance is used, the output becomes: xxvvc
            (c: covariance xvxv).
        """
        i = int(self.t / self.dt)
        outputs[:self.n_task_dims] = self.y[i]
        outputs[self.n_task_dims:2 * self.n_task_dims] = self.yd[i]
        if self.use_covar:
            outputs[2 * self.n_task_dims:] = self.covars[i].flatten()

    def step(self):
        """Compute desired position, velocity and acceleration."""

        # for computational reasons the step function just updates the time.
        # the trajectory is calculated beforehand.
        if self.n_task_dims == 0:
            return

        if self.t == self.last_t:
            self.last_t = -1.0
        else:
            self.last_t = self.t
            self.t += self.dt

        if not self._params_set:
            raise Exception("ProMP weights not set")

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
            Number of promp weights
        """
        random_variables = len(self.data.mean_)
        if self.learn_covariance:
            correlation_coefficients = (
                len(self.data.covariance_) - random_variables) / 2
            return 2 * random_variables + correlation_coefficients
        else:
            return random_variables

    def get_params(self):
        """Get current weights.

        Returns
        -------
        params : array-like, shape = (n_params,)
            Current weights
        """
        random_variables = self.data.mean_
        if self.learn_covariance:
            cov = np.array(self.data.covariance_).reshape(
                len(random_variables), len(random_variables))
            Dinv = np.linalg.inv(np.sqrt(np.diag(np.diag(cov))))
            cor = Dinv.dot(cov.dot(Dinv))

            return random_variables + np.sqrt(np.diag(cov)).tolist() + \
                np.arctanh(cor[np.tril_indices(len(random_variables), k=-1)]) \
                .tolist()
        else:
            return random_variables

    def set_params(self, params):
        """Set new weights.

        Parameters
        ----------
        params : array-like, shape = (n_params,)
            New weights
        """
        self.data.mean_ = params[:len(self.data.mean_)]
        if self.learn_covariance:
            D = np.diag(
                np.square(
                    np.array(params[len(self.data.mean_):2 *
                                    len(self.data.mean_)])))
            cor = np.identity(len(self.data.mean_))
            cor[np.tril_indices(len(self.data.mean_), k=-1)] = np.tanh(
                params[2 * len(self.data.mean_):])
            cor[np.triu_indices(len(self.data.mean_), k=1)] = np.tanh(
                params[2 * len(self.data.mean_):][::-1])
            self.data.covariance_ = D.dot(cor.dot(D)).flatten().tolist()
        self.y, self.yd, self.covars = self.trajectory()
        self._params_set = True

    def reset(self):
        """Reset promp."""
        if self.x0 is None:
            self.last_y = np.zeros(self.n_task_dims)
        else:
            self.last_y = np.copy(self.x0)
        self.last_yd = np.copy(self.x0d)
        self.last_ydd = np.copy(self.x0dd)

        self.last_t = 0.0
        self.t = 0.0

    def imitate(self,
                X,
                Xd=None,
                Xdd=None,
                alpha=0.0,
                allow_final_velocity=True):
        """Learn weights of the promp from demonstrations.

        Parameters
        ----------
        X : array, shape (n_task_dims, n_steps, n_demos)
            The demonstrated trajectories to be imitated.

        Xd : array, shape (n_task_dims, n_steps, n_demos), optional
            Velocities of the demonstrated trajectories.

        allow_final_velocity : bool, optional (default: True)
            Allow the final velocity to be greater than 0
        """
        if Xd is not None:
            warnings.warn("Xd is deprecated")
        if Xdd is not None:
            warnings.warn("Xdd is deprecated")
        if not allow_final_velocity:
            warnings.warn("allow_final_velocity is deprecated")

        y = X.transpose(2, 1, 0)
        x = np.arange(0, self.execution_time + self.dt * 0.1, self.dt)
        assert (x.shape[0] == y.shape[1])
        sizes_ = np.array([x.shape[0]], float).repeat(y.shape[0]).flatten()
        x_ = np.tile(x, y.shape[0]).flatten()
        y_ = y.flatten()
        self.data.imitate(sizes_, x_, y_)
        self.y, self.yd, self.covars = self.trajectory()
        self._params_set = True

    def trajectory(self):
        """Generate trajectory represented by the promp in open loop.

        The function can be used for debugging purposes.

        Returns
        -------
        X : array, shape (n_steps, n_task_dims)
            Positions

        Xd : array, shape (n_steps, n_task_dims)
            Velocities

        """

        # ret = promp.TrajectoryData(self.n_features,self.n_task_dims,True,
        # self.overlap) #TODO make param
        # self.data.sample_trajectory_data(ret)

        x = np.arange(0, self.execution_time + self.dt * 0.1, self.dt)

        means = np.empty((self.n_task_dims * 2 * len(x)))
        covars = np.empty(((self.n_task_dims * 2)**2) * len(x))

        self.data.get_values(x, means, covars)

        means = means.reshape((self.n_task_dims * 2, -1)).transpose(1, 0)
        covars = covars.reshape((-1, self.n_task_dims * 2,
                                 self.n_task_dims * 2))

        Y = means[:, ::2]
        Yd = means[:, 1::2]

        return Y, Yd, covars

    def distribution(self):
        """Generate a trajectory's mean and covariance represented by the promp in open loop.

        The function can be used for debugging purposes.

        Returns
        -------
        means : array, shape (n_steps, n_inputs)
            Contains positions, velocities in that order.
            Each type is stored contiguously, i.e. for n_task_dims=2 the order
            would be: xvxv (x: position, v: velocity).


        covariances : array, shape (n_steps, n_inputs,n_inputs)
            Contains positions, velocities in that order.
            Each type is stored contiguously, i.e. for n_task_dims=2 the order
            would be: xvxv
                      v
                      x
                      v    (x: position, v: velocity).


        """

        x = np.arange(0, self.execution_time + 0.000001, self.dt)

        means = np.empty((self.n_inputs * len(x)))
        covars = np.empty((self.n_inputs**2) * len(x))
        self.data.get_values(x, means, covars)
        means = means.reshape((self.n_inputs, -1)).transpose(1, 0)
        covars = covars.reshape((-1, self.n_inputs, self.n_inputs))

        return means, covars

    save = save_promp_model

    def save_config(self, filename):
        """Save promp configuration.

        Parameters
        ----------
        filename : string
            Name of YAML file
        """
        config = {}
        config["name"] = self.name
        config["promp_execution_time"] = self.execution_time
        config["promp_startPosition"] = self.x0.tolist()
        config["promp_startVelocity"] = self.x0d.tolist()
        config["promp_endPosition"] = self.g.tolist()
        config["promp_endVelocity"] = self.gd.tolist()

        config_content = StringIO()
        yaml.dump(config, config_content)
        with open(filename, "w") as f:
            f.write("---\n")
            f.write(config_content.getvalue())
            f.write("...\n")
        config_content.close()

    def load_config(self, filename):
        """Load promp configuration.

        Parameters
        ----------
        filename : string
            Name of YAML file
        """
        config = yaml.load(open(filename, "r"))
        self.execution_time = config["promp_execution_time"]
        self.x0 = np.array(config["promp_startPosition"], dtype=np.float)
        self.x0d = np.array(config["promp_startVelocity"], dtype=np.float)
        self.g = np.array(config["promp_endPosition"], dtype=np.float)
        self.gd = np.array(config["promp_endVelocity"], dtype=np.float)


def plot_covariance(ax, means, covariances, nstd=2):
    from matplotlib.patches import Ellipse

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    cov = np.empty((2, 2))
    for k in range(0, len(means)):
        cov[0, 0] = covariances[k, 0, 0]
        cov[0, 1] = covariances[k, 0, 2]
        cov[1, 0] = covariances[k, 2, 0]
        cov[1, 1] = covariances[k, 2, 2]

        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * nstd * np.sqrt(vals)

        ell = Ellipse(
            xy=means[k],
            width=width,
            height=height,
            angle=theta,
            alpha=1,
            edgecolor="none",
            facecolor="grey")
        ax.add_patch(ell)
