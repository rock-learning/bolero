from libcpp cimport bool


cdef extern from "../src/Dmp.h" namespace "Dmp":
    double calculateAlpha(double goal_z, double goal_t, double start_t) except +

cdef extern from "../src/Dmp.h" namespace "Dmp":
    void initializeRbf(double * widths, int num_widths, double * centers, int num_centers, double goal_t, double start_t, double overlap, double alpha) except +

cdef extern from "../src/Dmp.h" namespace "Dmp":
    void imitate(double * T, int num_T, double * Y, int num_steps, int num_task_dims, double * weights, int num_weights_per_dim, int num_weight_dims, double * widths, int num_widths, double * centers, int num_centers, double regularization_coefficient, double alpha_y, double beta_y, double alpha_z, bool allow_final_velocity) except +

cdef extern from "../src/Dmp.h" namespace "Dmp":
    void dmpStep(double last_t, double t, double * last_y, int num_last_y, double * last_yd, int num_last_yd, double * last_ydd, int num_last_ydd, double * y, int num_y, double * yd, int num_yd, double * ydd, int num_ydd, double * goal_y, int num_goal_y, double * goal_yd, int num_goal_yd, double * goal_ydd, int num_goal_ydd, double * start_y, int num_start_y, double * start_yd, int num_start_yd, double * start_ydd, int num_start_ydd, double goal_t, double start_t, double * weights, int num_weights_per_dim, int num_weight_dims, double * widths, int num_widths, double * centers, int num_centers, double alpha_y, double beta_y, double alpha_z, double integration_dt) except +

cdef extern from "../src/Dmp.h" namespace "Dmp":
    void quaternionImitate(double * T, int num_T, double * R, int num_steps, int num_task_dims, double * weights, int num_weights_per_dim, int num_weight_dims, double * widths, int num_widths, double * centers, int num_centers, double regularization_coefficient, double alpha_r, double beta_r, double alpha_z, bool allow_final_velocity) except +

cdef extern from "../src/Dmp.h" namespace "Dmp":
    void quaternionDmpStep(double last_t, double t, double * last_r, int num_last_r, double * last_rd, int num_last_rd, double * last_rdd, int num_last_rdd, double * r, int num_r, double * rd, int num_rd, double * rdd, int num_rdd, double * goal_r, int num_goal_r, double * goal_rd, int num_goal_rd, double * goal_rdd, int num_goal_rdd, double * start_r, int num_start_r, double * start_rd, int num_start_rd, double * start_rdd, int num_start_rdd, double goal_t, double start_t, double * weights, int num_weights_per_dim, int num_weight_dims, double * widths, int num_widths, double * centers, int num_centers, double alpha_r, double beta_r, double alpha_z, double integration_dt) except +

cdef extern from "../src/Dmp.h" namespace "Dmp::internal":
    void compute_gradient(double * _in, int num_in_steps, int num_in_dims, double * out, int num_out_steps, int num_out_dims, double * time, int num_time, bool allow_final_velocity) except +

cdef extern from "../src/Dmp.h" namespace "Dmp::internal":
    void compute_quaternion_gradient(double * _in, int num_in_steps, int num_in_dims, double * out, int num_out_steps, int num_out_dims, double * time, int num_time, bool allow_final_velocity) except +
