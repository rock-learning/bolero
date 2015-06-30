"""Rigid Body Dynamical Movement Primitives (Python wrapper for C++ implementation)."""

cimport numpy as np
import numpy as np
cimport cbindings as cb
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.string cimport string


cdef class RbDMP:
    """Rigid Body Dynamical Movement Primitives in C++.

    This is the Cython wrapper for the C++ implementation of rigid body DMPs.
    """

    cdef cb.RigidBodyDmp *thisptr
    cdef int n_features
    cdef int n_phases
    cdef double alpha
    cdef double beta
    cdef double dt
    cdef double T
    cdef double cs_alpha

    def __cinit__(self, np.ndarray[double, ndim=1] rbf_centers, np.ndarray[double, ndim=1] rbf_widths,
                    np.ndarray[double, ndim=2] ft_weights = None, execution_time=1.0, dt=0.01,
                    s_num_phases=0.01, alpha=25.0, beta=6.25):
        self.thisptr = new cb.RigidBodyDmp(NULL)
        self.n_features = rbf_centers.shape[0]
        self.n_phases = int(execution_time / dt) + 1
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.T = execution_time

        yaml = """---
name: ''
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
..."""

        self.cs_alpha = cb.calculateAlpha(s_num_phases, self.n_phases)
        if ft_weights is None:
            ft_weights = np.zeros((6, rbf_centers.shape[0]), order="F")

        yaml = yaml.format(rbf_centers=rbf_centers.tolist(), rbf_widths=rbf_widths.tolist(), ft_weights=ft_weights.tolist(),
                           ts_alpha_z=alpha, ts_beta_z=beta, ts_tau=execution_time, ts_dt = dt, cs_execution_time=execution_time,
                           cs_alpha=self.cs_alpha, cs_dt=dt)

        cdef char* yaml_ptr = yaml
        if not self.thisptr.initializeYaml(string(yaml_ptr)):
            raise Exception("DMP initialization failed")
            



    def __dealloc__(self):
        del self.thisptr

    def get_alpha(self):
        return self.cs_alpha

    def configure(self, start_pos, start_vel, start_acc, start_rot,
                  start_rot_vel, end_pos, end_vel, end_acc, end_rot):
        
        yaml = """---
name: ''
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
..."""        

        yaml = yaml.format(start_pos=start_pos.tolist(), end_pos=end_pos.tolist(),
                           start_vel=start_vel.tolist(), end_vel=end_vel.tolist(),
                           start_acc=start_acc.tolist(), end_acc=end_acc.tolist(),
                           start_rot=start_rot.tolist(), end_rot=end_rot.tolist(),
                           start_rot_vel=start_rot_vel.tolist(), execution_time=self.T)
        cdef char* yaml_ptr = yaml
        if not self.thisptr.configureYaml(string(yaml_ptr)):
            raise Exception("DMP configuration failed")



    def determine_forces(self, np.ndarray[double, ndim=2] X, *args, **kwargs):
        '''
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

        kwargs:Only exists to stay compatible with an old interface. Will be ignored. DO NOT USE

        Return:
        -------
        A 6xN matrix containing the forces.
        Each row contains the forces for one dimension.
        Note that one dimension is missing because rotation velocities are 3-dimensional.
        '''

        assert(X.shape[0] == 7)
        assert(X.shape[1] == self.n_phases)

        #the c++ interfaces requires that the arrays are stored in column major order
        if not X.flags["F_CONTIGUOUS"]:
            X = np.asfortranarray(X)

        cdef np.ndarray[double, ndim=2, mode="fortran"] forces = np.ndarray((6, self.n_phases), order="F")
        cb.determineForces(&X[0,0], 7, self.n_phases, &forces[0,0], 6, self.n_phases,
                           self.T, self.dt, self.alpha, self.beta)
        return forces



    @classmethod
    def calculate_centers(cls, s_num_phases, execution_time, dt, num_centers, overlap):
        """
        Calculates the centers and widths needed to configure a RbDmp.

        Parameters:
        -----------
        s_num_phases: The value of the last phase
        overlap: Overlap between the basis functions in percent ]0..1[

        Return:
        -------
        (centers, widths)
        """
        cdef np.ndarray[double, ndim=1, mode="fortran"] centers = np.ndarray((num_centers), order="F")
        cdef np.ndarray[double, ndim=1, mode="fortran"] widths = np.ndarray((num_centers), order="F")

        cb.calculateCenters(s_num_phases, execution_time, dt, num_centers,overlap, &centers[0],
                          &widths[0])
        return (centers, widths)



    @classmethod
    def _determine_forces(cls, np.ndarray[double, ndim=2] positions, np.ndarray[double, ndim=2] rotations,
                         dt, execution_time, alpha_z=25.0, beta_z=6.25):
        """
        Parameters
        ----------
        positions: 3xN array containing the positions. Each column should contain one 3-dimensional position.
        rotations: 4xN array containing the rotations. Each column should contain one quaternion.
                   Quaternion encoding: Row 0: w
                                        Row 1: x
                                        Row 2: y
                                        Row 3: z
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

        cdef np.ndarray[double, ndim=2, mode="fortran"] forces = np.ndarray((6, num_phases), order="F")

        cb.determineForces(&positions[0,0], 3, num_phases, &rotations[0,0], 4, num_phases, &forces[0,0], 6, num_phases,
                           execution_time, dt, alpha_z, beta_z)
        return forces

    def can_step(self):
        return self.thisptr.canStep()

    def execute_step(self, np.ndarray[double, ndim=1] position, np.ndarray[double, ndim=1] velocity,
                     np.ndarray[double, ndim=1] acceleration, np.ndarray[double, ndim=1] rotation):
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
        cdef np.ndarray[double, ndim=1, mode="fortran"] act = np.ndarray(self.n_features, order="F")
        self.thisptr.getActivations(s, normalized, &act[0], act.shape[0])
        return act

    def set_weights(self, w):
        assert(w.ndim == 1 or w.ndim == 2)
        cdef int n_task_dims = w.shape[0]
        cdef np.ndarray[double, ndim=2, mode="fortran"] wc
        if w.ndim == 1:
            n_task_dims = w.shape[0] / self.n_features
            wc = np.asfortranarray(w.reshape(n_task_dims, self.n_features))
        elif w.ndim == 2:
            wc = np.asfortranarray(w)

        assert(n_task_dims == 6)
        self.thisptr.setWeights(&wc[0,0], n_task_dims, self.n_features)

    def get_phases(self):
        cdef np.ndarray[double, ndim=1, mode="fortran"] s = np.ndarray(self.n_phases, order="F")
        self.thisptr.getPhases(&s[0], self.n_phases)
        return s
