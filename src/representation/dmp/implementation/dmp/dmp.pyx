from libcpp cimport bool
cimport numpy as np
import numpy as np
cimport _declarations as cpp


cpdef phase(double t, double alpha, double goal_t, double start_t):
    """Determine phase value that corresponds to the current time in the DMP.

    \param t current time, note that t is allowed to be outside of the range
    [start_t, goal_t]
    \param alpha constant that defines the decay rate of the phase variable
    \param goal_t time at the end of the DMP
    \param start_t time at the start of the DMP
    \return phase value (z)
    """
    cdef double cpp_t = t
    cdef double cpp_alpha = alpha
    cdef double cpp_goal_t = goal_t
    cdef double cpp_start_t = start_t
    cdef double result = cpp.phase(cpp_t, cpp_alpha, cpp_goal_t, cpp_start_t)
    return result


cpdef calculate_alpha(double goal_z, double goal_t, double start_t):
    """Compute decay rate of phase variable so that a desired phase is reached in
    the end.

    \param goal_z desired phase value
    \param goal_t time at the end of the DMP
    \param start_t time at the start of the DMP
    """
    cdef double cpp_goal_z = goal_z
    cdef double cpp_goal_t = goal_t
    cdef double cpp_start_t = start_t
    cdef double result = cpp.calculateAlpha(cpp_goal_z, cpp_goal_t, cpp_start_t)
    return result


cpdef initialize_rbf(np.ndarray[double, ndim=1] widths, np.ndarray[double, ndim=1] centers, double goal_t, double start_t, double overlap, double alpha):
    """Initialize radial basis functions.

    \param widths widths of the RBFs, will be initialized
    \param num_widths number of RBFs
    \param centers centers of the RBFs, will be initialized
    \param num_centers number of RBFs
    \param goal_t time at the end of the DMP
    \param start_t time at the start of the DMP
    \param overlap value of each RBF at the center of the next RBF
    \param alpha decay rate of the phase variable (default: 25.0 / 3.0)
    """
    cdef double cpp_goal_t = goal_t
    cdef double cpp_start_t = start_t
    cdef double cpp_overlap = overlap
    cdef double cpp_alpha = alpha
    cpp.initializeRbf(&widths[0], widths.shape[0], &centers[0], centers.shape[0], cpp_goal_t, cpp_start_t, cpp_overlap, cpp_alpha)

cpdef imitate(np.ndarray[double, ndim=1] T, np.ndarray[double, ndim=2] Y, np.ndarray[double, ndim=2] weights, np.ndarray[double, ndim=1] widths, np.ndarray[double, ndim=1] centers, double regularization_coefficient, double alpha_y, double beta_y, double alpha_z, bool allow_final_velocity):
    """Represent trajectory as DMP.

    \note The final velocity will be calculated by numeric differentiation
    from the data if allow_final_velocity is true. Otherwise we will assume
    the final velocity to be zero. To reproduce the trajectory as closely as
    possible, set the initial acceleration and velocity during execution to
    zero, the final acceleration to zero and the final velocity to the value
    that has been used during imitation.

    \param T time for each step of the trajectory
    \param num_T number of steps
    \param Y positions, contains num_T * num_dimensions entries in row-major
    order, i.e. the first position is located at the first num_dimensions
    entries of the array
    \param num_steps number of steps
    \param num_task_dims number of dimensions
    \param weights weights that reproduce the trajectory (will be updated)
    \param num_weights_per_dim number of features per dimension
    \param num_weight_dims number of dimensions
    \param widths widths of the radial basis functions (shared among DOFs)
    \param num_widths number of RBFs
    \param centers centers of the radial basis functions (shared among DOFs)
    \param num_centers number of RBFs
    \param regularization_coefficient can be set to solve instable problems
    where there are more weights that have to be learned than samples
    in the demonstrated trajectory (default: 1e-10)
    \param alpha_y constant that has to be set for critical damping (default: 25)
    \param beta_y constant that has to be set for critical damping (default: 25 / 4.0)
    \param alpha_z decay rate of the phase variable (default: 25.0 / 3.0)
    \param allow_final_velocity compute the final velocity from the data,
    otherwise we will assume it to be zero
    """
    cdef double cpp_regularization_coefficient = regularization_coefficient
    cdef double cpp_alpha_y = alpha_y
    cdef double cpp_beta_y = beta_y
    cdef double cpp_alpha_z = alpha_z
    cdef bool cpp_allow_final_velocity = allow_final_velocity
    cpp.imitate(&T[0], T.shape[0], &Y[0, 0], Y.shape[0], Y.shape[1], &weights[0, 0], weights.shape[0], weights.shape[1], &widths[0], widths.shape[0], &centers[0], centers.shape[0], cpp_regularization_coefficient, cpp_alpha_y, cpp_beta_y, cpp_alpha_z, cpp_allow_final_velocity)

cpdef dmp_step(double last_t, double t, np.ndarray[double, ndim=1] last_y, np.ndarray[double, ndim=1] last_yd, np.ndarray[double, ndim=1] last_ydd, np.ndarray[double, ndim=1] y, np.ndarray[double, ndim=1] yd, np.ndarray[double, ndim=1] ydd, np.ndarray[double, ndim=1] goal_y, np.ndarray[double, ndim=1] goal_yd, np.ndarray[double, ndim=1] goal_ydd, np.ndarray[double, ndim=1] start_y, np.ndarray[double, ndim=1] start_yd, np.ndarray[double, ndim=1] start_ydd, double goal_t, double start_t, np.ndarray[double, ndim=2] weights, np.ndarray[double, ndim=1] widths, np.ndarray[double, ndim=1] centers, double alpha_y, double beta_y, double alpha_z, double integration_dt):
    """Execute one step of the DMP.

    source: http://ijr.sagepub.com/content/32/3/263.full.pdf

    \param last_t time of last step (should equal t initially)
    \param t current time
    \param last_y last position
    \param num_last_y number of dimensions
    \param last_yd last velocity
    \param num_last_yd number of dimensions
    \param last_ydd last acceleration
    \param num_last_ydd number of dimensions
    \param y current position (will be updated)
    \param num_y number of dimensions
    \param yd velocity (will be updated)
    \param num_yd number of dimensions
    \param ydd acceleration (will be updated)
    \param num_ydd number of dimensions
    \param goal_y goal position
    \param num_goal_y number of dimensions
    \param goal_yd goal velocity
    \param num_goal_yd number of dimensions
    \param goal_ydd goal acceleration
    \param num_goal_ydd number of dimensions
    \param start_y start position
    \param num_start_y number of dimensions
    \param start_yd start velocity
    \param num_start_yd number of dimensions
    \param start_ydd start acceleration
    \param num_start_ydd number of dimensions
    \param goal_t time at the end of the DMP
    \param start_t time at the start of the DMP
    \param weights weights of the forcing term
    \param num_weights_per_dim number of features per dimension
    \param num_weight_dims number of dimensions
    \param widths widths of the radial basis functions (shared among DOFs)
    \param num_widths number of RBFs
    \param centers centers of the radial basis functions (shared among DOFs)
    \param num_centers number of RBFs
    \param alpha_y constant that has to be set for critical damping (default: 25)
    \param beta_y constant that has to be set for critical damping (default: 25 / 4.0)
    \param alpha_z decay rate of the phase variable (default: 25.0 / 3.0)
    \param integration_dt temporal step-size that will be used to integrate the
    velocity and position of the trajectory from the acceleration,
    smaller values will require more computation but will reproduce the
    demonstration more accurately
    """
    cdef double cpp_last_t = last_t
    cdef double cpp_t = t
    cdef double cpp_goal_t = goal_t
    cdef double cpp_start_t = start_t
    cdef double cpp_alpha_y = alpha_y
    cdef double cpp_beta_y = beta_y
    cdef double cpp_alpha_z = alpha_z
    cdef double cpp_integration_dt = integration_dt
    cpp.dmpStep(cpp_last_t, cpp_t, &last_y[0], last_y.shape[0], &last_yd[0], last_yd.shape[0], &last_ydd[0], last_ydd.shape[0], &y[0], y.shape[0], &yd[0], yd.shape[0], &ydd[0], ydd.shape[0], &goal_y[0], goal_y.shape[0], &goal_yd[0], goal_yd.shape[0], &goal_ydd[0], goal_ydd.shape[0], &start_y[0], start_y.shape[0], &start_yd[0], start_yd.shape[0], &start_ydd[0], start_ydd.shape[0], cpp_goal_t, cpp_start_t, &weights[0, 0], weights.shape[0], weights.shape[1], &widths[0], widths.shape[0], &centers[0], centers.shape[0], cpp_alpha_y, cpp_beta_y, cpp_alpha_z, cpp_integration_dt)

cpdef quaternion_imitate(np.ndarray[double, ndim=1] T, np.ndarray[double, ndim=2] R, np.ndarray[double, ndim=2] weights, np.ndarray[double, ndim=1] widths, np.ndarray[double, ndim=1] centers, double regularization_coefficient, double alpha_r, double beta_r, double alpha_z, bool allow_final_velocity):
    """Represent trajectory as quaternion DMP.

    \note The final velocity will be calculated by numeric differentiation
    from the data if allow_final_velocity is true. Otherwise we will assume
    the final velocity to be zero. To reproduce the trajectory as closely as
    possible, set the initial acceleration and velocity during execution to
    zero, the final acceleration to zero and the final velocity to the value
    that has been used during imitation.

    \param T time for each step of the trajectory
    \param num_T number of steps
    \param R rotations, contains num_T * 4 entries in row-major order, i.e.
    the first quaternion is located at the first 4 entries of the array
    \param num_steps number of steps
    \param num_task_dims should be 4
    \param weights weights that reproduce the trajectory (will be updated)
    \param num_weights_per_dim number of features per dimension
    \param num_weight_dims should be 3
    \param widths widths of the radial basis functions (shared among DOFs)
    \param num_widths number of RBFs
    \param centers centers of the radial basis functions (shared among DOFs)
    \param num_centers number of RBFs
    \param regularization_coefficient can be set to solve instable problems
    where there are more weights that have to be learned than samples
    in the demonstrated trajectory (default: 1e-10)
    \param alpha_r constant that has to be set for critical damping (default: 25)
    \param beta_r constant that has to be set for critical damping (default: 25 / 4.0)
    \param alpha_z decay rate of the phase variable (default: 25.0 / 3.0)
    \param allow_final_velocity compute the final velocity from the data,
    otherwise we will assume it to be zero
    """
    cdef double cpp_regularization_coefficient = regularization_coefficient
    cdef double cpp_alpha_r = alpha_r
    cdef double cpp_beta_r = beta_r
    cdef double cpp_alpha_z = alpha_z
    cdef bool cpp_allow_final_velocity = allow_final_velocity
    cpp.quaternionImitate(&T[0], T.shape[0], &R[0, 0], R.shape[0], R.shape[1], &weights[0, 0], weights.shape[0], weights.shape[1], &widths[0], widths.shape[0], &centers[0], centers.shape[0], cpp_regularization_coefficient, cpp_alpha_r, cpp_beta_r, cpp_alpha_z, cpp_allow_final_velocity)

cpdef quaternion_dmp_step(double last_t, double t, np.ndarray[double, ndim=1] last_r, np.ndarray[double, ndim=1] last_rd, np.ndarray[double, ndim=1] last_rdd, np.ndarray[double, ndim=1] r, np.ndarray[double, ndim=1] rd, np.ndarray[double, ndim=1] rdd, np.ndarray[double, ndim=1] goal_r, np.ndarray[double, ndim=1] goal_rd, np.ndarray[double, ndim=1] goal_rdd, np.ndarray[double, ndim=1] start_r, np.ndarray[double, ndim=1] start_rd, np.ndarray[double, ndim=1] start_rdd, double goal_t, double start_t, np.ndarray[double, ndim=2] weights, np.ndarray[double, ndim=1] widths, np.ndarray[double, ndim=1] centers, double alpha_r, double beta_r, double alpha_z, double integration_dt):
    """Execute one step of the Quaternion DMP.

    source: http://ieeexplore.ieee.org/document/6907291/?arnumber=6907291

    \param last_t time of last step (should equal t initially)
    \param t current time
    \param last_r last rotation
    \param num_last_r should be 4
    \param last_rd last rotational velocity
    \param num_last_rd should be 3
    \param last_rdd last rotational acceleration
    \param num_last_rdd should be 3
    \param r current rotation (will be updated)
    \param num_r should be 4
    \param rd rotational velocity (will be updated)
    \param num_rd should be 3
    \param rdd rotational acceleration (will be updated)
    \param num_rdd should be 3
    \param goal_r final rotation
    \param num_goal_r should be 4
    \param goal_rd final rotational velocity
    \param num_goal_rd should be 3
    \param goal_rdd final rotational acceleration
    \param num_goal_rdd should be 3
    \param start_r first rotation
    \param num_start_r should be 4
    \param start_rd first rotational velocity
    \param num_start_rd should be 3
    \param start_rdd first rotational acceleration
    \param num_start_rdd should be 3
    \param goal_t time at the end of the DMP
    \param start_t time at the start of the DMP
    \param weights weights of the forcing term
    \param num_weights_per_dim number of features per dimension
    \param num_weight_dims should be 3
    \param widths widths of the radial basis functions (shared among DOFs)
    \param num_widths number of RBFs
    \param centers centers of the radial basis functions (shared among DOFs)
    \param num_centers number of RBFs
    \param alpha_r constant that has to be set for critical damping (default: 25)
    \param beta_r constant that has to be set for critical damping (default: 25 / 4.0)
    \param alpha_z decay rate of the phase variable (default: 25.0 / 3.0)
    \param integration_dt temporal step-size that will be used to integrate the
    velocity and position of the trajectory from the acceleration,
    smaller values will require more computation but will reproduce the
    demonstration more accurately
    """
    cdef double cpp_last_t = last_t
    cdef double cpp_t = t
    cdef double cpp_goal_t = goal_t
    cdef double cpp_start_t = start_t
    cdef double cpp_alpha_r = alpha_r
    cdef double cpp_beta_r = beta_r
    cdef double cpp_alpha_z = alpha_z
    cdef double cpp_integration_dt = integration_dt
    cpp.quaternionDmpStep(cpp_last_t, cpp_t, &last_r[0], last_r.shape[0], &last_rd[0], last_rd.shape[0], &last_rdd[0], last_rdd.shape[0], &r[0], r.shape[0], &rd[0], rd.shape[0], &rdd[0], rdd.shape[0], &goal_r[0], goal_r.shape[0], &goal_rd[0], goal_rd.shape[0], &goal_rdd[0], goal_rdd.shape[0], &start_r[0], start_r.shape[0], &start_rd[0], start_rd.shape[0], &start_rdd[0], start_rdd.shape[0], cpp_goal_t, cpp_start_t, &weights[0, 0], weights.shape[0], weights.shape[1], &widths[0], widths.shape[0], &centers[0], centers.shape[0], cpp_alpha_r, cpp_beta_r, cpp_alpha_z, cpp_integration_dt)

cpdef compute_gradient(np.ndarray[double, ndim=2] _in, np.ndarray[double, ndim=2] out, np.ndarray[double, ndim=1] time, bool allow_final_velocity):
    cdef bool cpp_allow_final_velocity = allow_final_velocity
    cpp.compute_gradient(&_in[0, 0], _in.shape[0], _in.shape[1], &out[0, 0], out.shape[0], out.shape[1], &time[0], time.shape[0], cpp_allow_final_velocity)

cpdef compute_quaternion_gradient(np.ndarray[double, ndim=2] _in, np.ndarray[double, ndim=2] out, np.ndarray[double, ndim=1] time, bool allow_final_velocity):
    cdef bool cpp_allow_final_velocity = allow_final_velocity
    cpp.compute_quaternion_gradient(&_in[0, 0], _in.shape[0], _in.shape[1], &out[0, 0], out.shape[0], out.shape[1], &time[0], time.shape[0], cpp_allow_final_velocity)
