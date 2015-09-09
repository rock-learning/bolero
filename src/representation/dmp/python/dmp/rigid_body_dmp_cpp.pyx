# distutils: language=c++
"""Rigid Body Dynamical Movement Primitives (wrapper for C++ implementation)."""

cimport numpy as np
import numpy as np
cimport cbindings as cb
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.string cimport string
import yaml


INIT_YAML = """---
name: 'PythonCSDMP'
rbf_centers: {rbf_centers}
rbf_widths: {rbf_widths}
ts_alpha_z: {ts_alpha_z}
ts_beta_z: {ts_beta_z}
ts_tau: {ts_tau}
ts_dt: {ts_dt}
cs_execution_time: {cs_execution_time}
cs_alpha: {cs_alpha}
cs_dt: {cs_dt}
ft_weights: {ft_weights}
...
"""


CONFIG_YAML = """---
name: 'PythonCSDMP'
startPosition: {start_pos}
endPosition: {end_pos}
startVelocity: {start_vel}
endVelocity: {end_vel}
startAcceleration: {start_acc}
endAcceleration: {end_acc}
startRotation: {start_rot}
endRotation: {end_rot}
startAngularVelocity: {start_rot_vel}
executionTime: {execution_time}
...
"""


cdef class RbDMP:
    """Rigid Body Dynamical Movement Primitives in C++.

    This is the Cython wrapper for the C++ implementation of rigid body DMPs.

    Parameters
    ----------
    execution_time : float, optional (default: 1)
        Execution time of the DMP in seconds.

    dt : float, optional (default: 0.01)
        Time between successive steps in seconds.

    n_features : int, optional (default: 50)
        Number of RBF features for each dimension of the DMP.

    s_num_phases : float, optional (default: 0.01)
        Value of the phase variable after 'execution_time'.

    overlap : float, optional (default: 0.8)
        Value of the RBFs at the phase value where they overlap.

    alpha : float, optional (default: 25)
        Constant that modifies the behavior of the spring-damper system.

    beta : float, optional (default: 6.25)
        Constant that modifies the behavior of the spring-damper system.
    """
    cdef cb.RigidBodyDmp *thisptr
    cdef int n_features
    cdef int n_phases
    cdef double alpha
    cdef double beta
    cdef double dt
    cdef double execution_time
    cdef double cs_alpha
    cdef bool initialized
    cdef string init_yaml
    cdef string config_yaml

    def __cinit__(self, execution_time=1.0, dt=0.01, n_features=50,
                  s_num_phases=0.01, overlap=0.8, alpha=25.0, beta=6.25):
        cdef np.ndarray[double, ndim=1] rbf_centers
        cdef np.ndarray[double, ndim=1] rbf_widths
        rbf_centers, rbf_widths = self.calculate_centers(
            s_num_phases, execution_time, dt, n_features, overlap)
        self.thisptr = new cb.RigidBodyDmp(NULL)
        self.n_features = n_features
        self.n_phases = int(execution_time / dt + 0.5) + 1
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.execution_time = execution_time

        self.initialized = False

        self.cs_alpha = cb.calculateAlpha(s_num_phases, self.n_phases)
        ft_weights = np.zeros((6, rbf_centers.shape[0]), order="F")

        self.init_yaml = INIT_YAML.format(
            rbf_centers=rbf_centers.tolist(), rbf_widths=rbf_widths.tolist(),
            ft_weights=ft_weights.tolist(), ts_alpha_z=alpha, ts_beta_z=beta,
            ts_tau=execution_time, ts_dt = dt, cs_execution_time=execution_time,
            cs_alpha=self.cs_alpha, cs_dt=dt)

        if not self.thisptr.initializeYaml(self.init_yaml):
            raise Exception("DMP initialization failed")

    def __dealloc__(self):
        del self.thisptr

    def reset(self):
        self.initialized = False
        w = self.get_weights()
        if not self.thisptr.initializeYaml(self.init_yaml):
            raise Exception("DMP initialization failed")
        if not self.thisptr.configureYaml(self.config_yaml):
            raise Exception("DMP configuration failed")
        self.set_weights(w)

    def get_alpha(self):
        return self.cs_alpha

    def configure(self, start_pos, start_vel, start_acc, start_rot,
                  start_rot_vel, end_pos, end_vel, end_acc, end_rot,
                  execution_time):
        self.execution_time = execution_time
        self.n_phases = int(self.execution_time / self.dt + 0.5) + 1
        self.config_yaml = CONFIG_YAML.format(
            start_pos=start_pos.tolist(), end_pos=end_pos.tolist(),
            start_vel=start_vel.tolist(), end_vel=end_vel.tolist(),
            start_acc=start_acc.tolist(), end_acc=end_acc.tolist(),
            start_rot=start_rot.tolist(), end_rot=end_rot.tolist(),
            start_rot_vel=start_rot_vel.tolist(),
            execution_time=execution_time)
        if not self.thisptr.configureYaml(self.config_yaml):
            raise Exception("DMP configuration failed")

    def determine_forces(self, np.ndarray[double, ndim=2] X, *args, **kwargs):
        """Determine forces for given demonstration.

        Parameters
        ----------
        X: A 7xN array containing the translational and rotational positions.
           Each column should contain one position. The rotation should be encoded as quaternion.
           Row 0: x
           Row 1: y
           Row 2: z
           Row 3: rot_w
           Row 4: rot_x
           Row 5: rot_y
           Row 6: rot_z

        args: Only exists to stay compatible with an old interface. Will be ignored. DO NOT USE

        kwargs: Only exists to stay compatible with an old interface. Will be ignored. DO NOT USE

        Return:
        -------
        A 6xN matrix containing the forces.
        Each row contains the forces for one dimension.
        Note that one dimension is missing because rotation velocities are 3-dimensional.
        """
        assert(X.shape[0] == 7)
        assert(X.shape[1] == self.n_phases,
               "%d trajectory samples given, expected %d phases"
               % (X.shape[1], self.n_phases))

        #the c++ interfaces requires that the arrays are stored in column major order
        if not X.flags["F_CONTIGUOUS"]:
            X = np.asfortranarray(X)

        cdef np.ndarray[double, ndim=2, mode="fortran"] forces = np.ndarray(
            (6, self.n_phases), order="F")
        cb.determineForces(&X[0, 0], 7, self.n_phases, &forces[0, 0], 6,
                           self.n_phases, self.execution_time, self.dt,
                           self.alpha, self.beta)
        return forces

    @classmethod
    def calculate_centers(cls, s_num_phases, execution_time, dt, num_centers,
                          overlap):
        """Calculates the centers and widths needed to configure a RbDmp.

        Parameters:
        -----------
        s_num_phases: The value of the last phase
        overlap: Overlap between the basis functions in percent ]0..1[

        Return:
        -------
        centers
        widths
        """
        cdef np.ndarray[double, ndim=1, mode="fortran"] centers = np.ndarray(
            (num_centers), order="F")
        cdef np.ndarray[double, ndim=1, mode="fortran"] widths = np.ndarray(
            (num_centers), order="F")

        cb.calculateCenters(s_num_phases, execution_time, dt, num_centers,
                            overlap, &centers[0], &widths[0])
        return centers, widths

    @classmethod
    def _determine_forces(cls, np.ndarray[double, ndim=2] positions,
                          np.ndarray[double, ndim=2] rotations, dt,
                          execution_time, alpha_z=25.0, beta_z=6.25):
        """Determine forces for given demonstration.

        Parameters
        ----------
        positions: 3xN array
            Contains the positions. Each column should contain one
            3-dimensional position.

        rotations: 4xN array
            Contains the rotations. Each column should contain one quaternion.
            Quaternion encoding: Row 0: w; Row 1: x; Row 2: y; Row 3: z
        """
        assert positions.shape[0] == 3
        assert rotations.shape[0] == 4
        assert positions.shape[1] == rotations.shape[1]
        num_phases = rotations.shape[1]

        #the c++ interfaces requires that the arrays are stored in column major order
        if not positions.flags["F_CONTIGUOUS"]:
            positions = np.asfortranarray(positions)
        if not rotations.flags["F_CONTIGUOUS"]:
            rotations = np.asfortranarray(rotations)

        cdef np.ndarray[double, ndim=2, mode="fortran"] forces = np.ndarray(
            (6, num_phases), order="F")

        cb.determineForces(&positions[0,0], 3, num_phases, &rotations[0,0], 4,
                           num_phases, &forces[0,0], 6, num_phases,
                           execution_time, dt, alpha_z, beta_z)
        return forces

    def can_step(self):
        return self.thisptr.canStep()

    def execute_step(
            self, np.ndarray[double, ndim=1] position,
            np.ndarray[double, ndim=1] velocity,
            np.ndarray[double, ndim=1] acceleration,
            np.ndarray[double, ndim=1] rotation):
        """
        Parameters
        ---------
        position: current position (x, y, z)
        velocity: current velocity (xd, yd, zd)
        acceleration: current acceleration (xdd, ydd, zdd)
        rotation: current rotation as quaternion (w, x, y, z)
        """
        assert position.shape[0] == 3
        assert velocity.shape[0] == 3
        assert acceleration.shape[0] == 3
        assert rotation.shape[0] == 4

        if not self.initialized:
            self.initialized = True
            return position, velocity, acceleration, rotation

        cdef np.ndarray[double, ndim=1] data = np.ndarray(13)
        data[0:3] = position
        data[3:6] = velocity
        data[6:9] = acceleration
        data[9:13] = rotation
        self.thisptr.setInputs(&data[0], 13)
        self.thisptr.step()
        self.thisptr.getOutputs(&data[0], 13)

        posOut = data[0:3]
        velOut = data[3:6]
        accOut = data[6:9]
        rotOut = data[9:13]

        return posOut, velOut, accOut, rotOut

    def get_activations(self, s, normalized=True):
        cdef np.ndarray[double, ndim=1, mode="fortran"] act = np.ndarray(
            self.n_features, order="F")
        self.thisptr.getActivations(s, normalized, &act[0], act.shape[0])
        return act

    def set_weights(self, w):
        if w.ndim != 1 and w.ndim != 2:
            raise ValueError("Expected weights with 1 or 2 dimensions, got %d"
                             % w.ndim)

        if w.size == 0:
            return

        cdef int n_task_dims = w.shape[0]
        cdef np.ndarray[double, ndim=2, mode="fortran"] wc
        if w.ndim == 1:
            n_task_dims = w.shape[0] / self.n_features
            wc = np.asfortranarray(w.reshape(n_task_dims, self.n_features))
        elif w.ndim == 2:
            wc = np.asfortranarray(w)

        assert(n_task_dims == 6)
        self.thisptr.setWeights(&wc[0, 0], n_task_dims, self.n_features)

    def get_weights(self):
        cdef np.ndarray[double, ndim=2, mode="fortran"] weights = np.ndarray(
            (6, self.n_features), order="F")
        self.thisptr.getWeights(&weights[0, 0], 6, self.n_features)
        return weights

    def get_phases(self):
        cdef np.ndarray[double, ndim=1, mode="fortran"] s = np.ndarray(
            self.n_phases, order="F")
        self.thisptr.getPhases(&s[0], self.n_phases)
        return s

    @classmethod
    def from_file(cls, filename):
        """Load DMP from YAML file.

        Parameters
        ----------
        filename : string
            Name of the YAML file that stores the DMP model.

        Returns
        -------
        dmp : RbDMP
            The corresponding DMP object.
        """
        dmp = RbDMP()
        del dmp.thisptr
        dmp.thisptr = new cb.RigidBodyDmp(NULL)

        init_yaml = open(filename, "r").read()
        init_dict = yaml.load(init_yaml)
        dmp.init_yaml = init_yaml
        if not dmp.thisptr.initializeYaml(init_yaml):
            raise Exception("DMP initialization failed")
        dmp.config_yaml = ""

        dmp.n_features = len(init_dict["rbf_centers"])
        dmp.alpha = init_dict["ts_alpha_z"]
        dmp.beta = init_dict["ts_beta_z"]
        dmp.dt = init_dict["ts_dt"]
        dmp.execution_time = init_dict["ts_tau"]
        dmp.n_phases = int(dmp.execution_time / dmp.dt + 0.5) + 1
        dmp.cs_alpha = init_dict["cs_alpha"]
        return dmp

    def save_model(self, filename):
        """Save DMP model in YAML file.

        Parameters
        ----------
        filename : string
            Name of the YAML file that stores the DMP model.
        """
        model = yaml.load(self.init_yaml)
        model["ft_weights"] = self.get_weights().tolist()
        yaml.dump(model, open(filename, "w"))
