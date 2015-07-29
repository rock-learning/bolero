# distutils: language=c++
"""Dynamical Movement Primitives (wrapper for C++ implementation)."""

cimport numpy as np
import numpy as np
cimport cbindings as cb
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.string cimport string


cdef class DMP:
    """Dynamical Movement Primitives in C++.

    This is the Cython wrapper for the C++ implementation of DMPs.

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

    integration_steps : int, optional (default: 4)
        Number of integration steps for each DMP step. Since we integrate
        numerically, more integration steps result in better approximations.
    """
    cdef cb.Dmp *thisptr
    cdef int n_phases
    cdef int n_features
    cdef bool initialized
    cdef bool run
    cdef np.ndarray weights
    cdef double execution_time
    cdef double dt
    cdef double alpha
    cdef double beta

    def __cinit__(self, execution_time=1.0, dt=0.01, n_features=50,
                  s_num_phases=0.01, overlap=0.8, alpha=25.0, beta=6.25,
                  integration_steps=4):
        self.n_phases = int(execution_time / dt) + 1
        self.n_features = n_features
        cs_alpha = cb.calculateAlpha(s_num_phases, self.n_phases)
        self.thisptr = new cb.Dmp(execution_time, cs_alpha, dt,
                                  self.n_features, overlap, alpha, beta,
                                  integration_steps)
        self.initialized = False
        self.run = True
        self.execution_time = execution_time
        self.dt = dt
        self.alpha = alpha
        self.beta = beta

    @classmethod
    def from_file(cls, filename):
        """Load DMP from YAML file.

        Parameters
        ----------
        filename : string
            Name of the YAML file that stores the DMP model.

        Returns
        -------
        dmp : DMP
            The corresponding DMP object.
        """
        cdef char* file = filename
        cdef char* name = ""
        cdef cb.DMPWrapper* wrapper = new cb.DMPWrapper()
        wrapper.init_from_yaml(string(file), string(name))
        cdef cb.DMPModel model = wrapper.generate_model()

        dmp = DMP()
        del dmp.thisptr
        dmp.thisptr = new cb.Dmp(wrapper.dmp())
        dmp.n_phases = int(model.ts_tau / model.ts_dt) + 1
        dmp.n_features = model.rbf_centers.size()
        return dmp

    def __dealloc__(self):
        del self.thisptr

    def set_metaparameters(self, keys, meta_parameters):
        for i in range(len(keys)):
            if keys[i] == "execution_time":
                self.thisptr.changeTime(meta_parameters[i])

    def determine_forces(self, np.ndarray[double, ndim=2] X,
                         np.ndarray[double, ndim=2] Xd=None,
                         np.ndarray[double, ndim=2] Xdd=None):
        cdef int n_task_dims = X.shape[0]
        cdef int n_phases = X.shape[1]
        
        # The c++ interfaces requires that the arrays are stored in column
        # major order
        if not X.flags["F_CONTIGUOUS"]:
            X = np.asfortranarray(X)

        cdef double* pXd = NULL
        cdef double* pXdd = NULL
        if Xd is not None:
            if not Xd.flags["F_CONTIGUOUS"]:
                Xd = np.asfortranarray(Xd)
            pXd = &Xd[0,0]

        if Xdd is not None:
            if not Xdd.flags["F_CONTIGUOUS"]:
                Xdd = np.asfortranarray(Xdd)
            pXdd = &Xdd[0,0]

        cdef np.ndarray[double, ndim=2, mode="fortran"] F = np.ndarray(
            (n_task_dims, n_phases), order="F")

        cb.determineForces(&X[0,0], pXd, pXdd, n_task_dims,
                           n_phases, &F[0,0], n_task_dims, n_phases,
                           self.execution_time, self.dt, self.alpha, self.beta) 

        return F

    def execute_step(self, np.ndarray[double] x,
                     np.ndarray[double] xd,
                     np.ndarray[double] xdd,
                     x0, g, gd=None, gdd=None):
        if not self.run:
            raise Exception("DMP has been executed, need reset.")
        if not self.initialized and g is None:
            raise ValueError("Goal is required for initialization!")

        cdef int n_task_dims = x.shape[0]
        assert xd.shape[0] == n_task_dims
        assert xdd.shape[0] == n_task_dims

        # The c++ interfaces expects contiguous arrays (row or column major
        # does not matter for 1d arrays)
        if not x.flags["F_CONTIGUOUS"]:
            x = np.asfortranarray(x)
        if not xd.flags["F_CONTIGUOUS"]:
            xd = np.asfortranarray(xd)
        if not xdd.flags["F_CONTIGUOUS"]:
            xdd = np.asfortranarray(xdd)

        cdef np.ndarray[double, mode="fortran"] gc
        cdef np.ndarray[double, mode="fortran"] gdc
        cdef np.ndarray[double, mode="fortran"] gddc

        if g is not None:
            gc = np.asfortranarray(g)
            if gc.shape[0] != n_task_dims:
                raise ValueError("Goal dimensions: %r, n_task_dims = %d"
                                 % (g.shape, n_task_dims))

            if gd is None:
                gdc = np.zeros_like(gc, order="F")
            else:
                gdc = np.asfortranarray(gd)
                assert gdc.shape[0] == n_task_dims
            if gdd is None:
                gddc = np.zeros_like(gc, order="F")
            else:
                gddc = np.asfortranarray(gdd)
                assert gddc.shape[0] == n_task_dims

        if not self.initialized:
            self.thisptr.initialize(&x[0], &xd[0], &xdd[0],
                                    &gc[0], &gdc[0], &gddc[0], n_task_dims)
            self.initialized = True
            return x, xd, xdd

        elif g is not None:  # Goal has changed
            self.thisptr.changeGoal(&gc[0], &gdc[0], &gddc[0], n_task_dims)

        self.run = self.thisptr.executeStep(&x[0], &xd[0], &xdd[0], n_task_dims)
        return x, xd, xdd

    def get_phases(self):
        cdef np.ndarray[double, ndim=1, mode="fortran"] s = np.ndarray(
            self.n_phases, order="F")
        self.thisptr.getPhases(&s[0], self.n_phases)
        return s

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
            n_task_dims /= self.n_features
            wc = np.asfortranarray(w.reshape(n_task_dims, self.n_features))
        elif w.ndim == 2:
            wc = np.asfortranarray(w)

        self.thisptr.setWeights(&wc[0, 0], n_task_dims, self.n_features)

    def get_weights(self):
        cdef int n_task_dims = self.thisptr.getTaskDimensions()
        if n_task_dims == 0:
            return None

        cdef np.ndarray[double, ndim=2, mode="fortran"] weights = np.ndarray(
            (n_task_dims, self.n_features), order="F")
        self.thisptr.getWeights(&weights[0, 0], n_task_dims, self.n_features)
        return weights

    def can_step(self):
        return self.run

    def reset(self):
        self.initialized = False
        self.run = True

    def get_num_features(self):
        return self.n_features

    def save_model(self, file_name):
        """Save DMP in YAML file.

        Parameters
        ----------
        file_name : string
            Name of the YAML file that stores the DMP model.
        """
        cdef char* file = file_name
        cdef cb.DMPWrapper* wrapper = new cb.DMPWrapper()
        wrapper.init_from_dmp(deref(self.thisptr))
        cdef cb.DMPModel model = wrapper.generate_model()
        model.to_yaml_file(string(file))
        del wrapper
