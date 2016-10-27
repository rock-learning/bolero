from libcpp cimport bool
cimport numpy as np
import numpy as np
cimport _declarations as cpp


cpdef calculate_alpha(double goal_z, double goal_t, double start_t):
    cdef double cpp_goal_z = goal_z
    cdef double cpp_goal_t = goal_t
    cdef double cpp_start_t = start_t
    cdef double result = cpp.calculateAlpha(cpp_goal_z, cpp_goal_t, cpp_start_t)
    return result

cpdef initialize_rbf(np.ndarray[double, ndim=1] widths, np.ndarray[double, ndim=1] centers, double goal_t, double start_t, double overlap, double alpha):
    cdef double cpp_goal_t = goal_t
    cdef double cpp_start_t = start_t
    cdef double cpp_overlap = overlap
    cdef double cpp_alpha = alpha
    cpp.initializeRbf(&widths[0], widths.shape[0], &centers[0], centers.shape[0], cpp_goal_t, cpp_start_t, cpp_overlap, cpp_alpha)

cpdef imitate(np.ndarray[double, ndim=1] T, np.ndarray[double, ndim=1] Y, np.ndarray[double, ndim=1] weights, np.ndarray[double, ndim=1] widths, np.ndarray[double, ndim=1] centers, double regularization_coefficient, double alpha_y, double beta_y, double alpha_z, bool allow_final_velocity):
    cdef double cpp_regularization_coefficient = regularization_coefficient
    cdef double cpp_alpha_y = alpha_y
    cdef double cpp_beta_y = beta_y
    cdef double cpp_alpha_z = alpha_z
    cdef bool cpp_allow_final_velocity = allow_final_velocity
    cpp.imitate(&T[0], T.shape[0], &Y[0], Y.shape[0], &weights[0], weights.shape[0], &widths[0], widths.shape[0], &centers[0], centers.shape[0], cpp_regularization_coefficient, cpp_alpha_y, cpp_beta_y, cpp_alpha_z, cpp_allow_final_velocity)

cpdef dmp_step(double last_t, double t, np.ndarray[double, ndim=1] last_y, np.ndarray[double, ndim=1] last_yd, np.ndarray[double, ndim=1] last_ydd, np.ndarray[double, ndim=1] y, np.ndarray[double, ndim=1] yd, np.ndarray[double, ndim=1] ydd, np.ndarray[double, ndim=1] goal_y, np.ndarray[double, ndim=1] goal_yd, np.ndarray[double, ndim=1] goal_ydd, np.ndarray[double, ndim=1] start_y, np.ndarray[double, ndim=1] start_yd, np.ndarray[double, ndim=1] start_ydd, double goal_t, double start_t, np.ndarray[double, ndim=1] weights, np.ndarray[double, ndim=1] widths, np.ndarray[double, ndim=1] centers, double alpha_y, double beta_y, double alpha_z, double integration_dt):
    cdef double cpp_last_t = last_t
    cdef double cpp_t = t
    cdef double cpp_goal_t = goal_t
    cdef double cpp_start_t = start_t
    cdef double cpp_alpha_y = alpha_y
    cdef double cpp_beta_y = beta_y
    cdef double cpp_alpha_z = alpha_z
    cdef double cpp_integration_dt = integration_dt
    cpp.dmpStep(cpp_last_t, cpp_t, &last_y[0], last_y.shape[0], &last_yd[0], last_yd.shape[0], &last_ydd[0], last_ydd.shape[0], &y[0], y.shape[0], &yd[0], yd.shape[0], &ydd[0], ydd.shape[0], &goal_y[0], goal_y.shape[0], &goal_yd[0], goal_yd.shape[0], &goal_ydd[0], goal_ydd.shape[0], &start_y[0], start_y.shape[0], &start_yd[0], start_yd.shape[0], &start_ydd[0], start_ydd.shape[0], cpp_goal_t, cpp_start_t, &weights[0], weights.shape[0], &widths[0], widths.shape[0], &centers[0], centers.shape[0], cpp_alpha_y, cpp_beta_y, cpp_alpha_z, cpp_integration_dt)

cpdef quaternion_imitate(np.ndarray[double, ndim=1] T, np.ndarray[double, ndim=1] R, np.ndarray[double, ndim=1] weights, np.ndarray[double, ndim=1] widths, np.ndarray[double, ndim=1] centers, double regularization_coefficient, double alpha_r, double beta_r, double alpha_z, bool allow_final_velocity):
    cdef double cpp_regularization_coefficient = regularization_coefficient
    cdef double cpp_alpha_r = alpha_r
    cdef double cpp_beta_r = beta_r
    cdef double cpp_alpha_z = alpha_z
    cdef bool cpp_allow_final_velocity = allow_final_velocity
    cpp.quaternionImitate(&T[0], T.shape[0], &R[0], R.shape[0], &weights[0], weights.shape[0], &widths[0], widths.shape[0], &centers[0], centers.shape[0], cpp_regularization_coefficient, cpp_alpha_r, cpp_beta_r, cpp_alpha_z, cpp_allow_final_velocity)

cpdef quaternion_dmp_step(double last_t, double t, np.ndarray[double, ndim=1] last_r, np.ndarray[double, ndim=1] last_rd, np.ndarray[double, ndim=1] last_rdd, np.ndarray[double, ndim=1] r, np.ndarray[double, ndim=1] rd, np.ndarray[double, ndim=1] rdd, np.ndarray[double, ndim=1] goal_r, np.ndarray[double, ndim=1] goal_rd, np.ndarray[double, ndim=1] goal_rdd, np.ndarray[double, ndim=1] start_r, np.ndarray[double, ndim=1] start_rd, np.ndarray[double, ndim=1] start_rdd, double goal_t, double start_t, np.ndarray[double, ndim=1] weights, np.ndarray[double, ndim=1] widths, np.ndarray[double, ndim=1] centers, double alpha_r, double beta_r, double alpha_z, double integration_dt):
    cdef double cpp_last_t = last_t
    cdef double cpp_t = t
    cdef double cpp_goal_t = goal_t
    cdef double cpp_start_t = start_t
    cdef double cpp_alpha_r = alpha_r
    cdef double cpp_beta_r = beta_r
    cdef double cpp_alpha_z = alpha_z
    cdef double cpp_integration_dt = integration_dt
    cpp.quaternionDmpStep(cpp_last_t, cpp_t, &last_r[0], last_r.shape[0], &last_rd[0], last_rd.shape[0], &last_rdd[0], last_rdd.shape[0], &r[0], r.shape[0], &rd[0], rd.shape[0], &rdd[0], rdd.shape[0], &goal_r[0], goal_r.shape[0], &goal_rd[0], goal_rd.shape[0], &goal_rdd[0], goal_rdd.shape[0], &start_r[0], start_r.shape[0], &start_rd[0], start_rd.shape[0], &start_rdd[0], start_rdd.shape[0], cpp_goal_t, cpp_start_t, &weights[0], weights.shape[0], &widths[0], widths.shape[0], &centers[0], centers.shape[0], cpp_alpha_r, cpp_beta_r, cpp_alpha_z, cpp_integration_dt)

cpdef compute_gradient(np.ndarray[double, ndim=1] _in, np.ndarray[double, ndim=1] out, np.ndarray[double, ndim=1] time, bool allow_final_velocity):
    cdef bool cpp_allow_final_velocity = allow_final_velocity
    cpp.compute_gradient(&_in[0], _in.shape[0], &out[0], out.shape[0], &time[0], time.shape[0], cpp_allow_final_velocity)

cpdef compute_quaternion_gradient(np.ndarray[double, ndim=1] _in, np.ndarray[double, ndim=1] out, np.ndarray[double, ndim=1] time, bool allow_final_velocity):
    cdef bool cpp_allow_final_velocity = allow_final_velocity
    cpp.compute_quaternion_gradient(&_in[0], _in.shape[0], &out[0], out.shape[0], &time[0], time.shape[0], cpp_allow_final_velocity)
