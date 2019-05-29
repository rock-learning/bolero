#pragma once

namespace Dmp {

/**
 * Determine phase value that corresponds to the current time in the DMP.
 * \param t current time, note that t is allowed to be outside of the range
 *          [start_t, goal_t]
 * \param alpha constant that defines the decay rate of the phase variable
 * \param goal_t time at the end of the DMP
 * \param start_t time at the start of the DMP
 * \return phase value (z)
 */
const double phase(
  const double t,
  const double alpha,
  const double goal_t,
  const double start_t
);

/**
 * Compute decay rate of phase variable so that a desired phase is reached in
 * the end.
 *
 * \param goal_z desired phase value
 * \param goal_t time at the end of the DMP
 * \param start_t time at the start of the DMP
 */
double calculateAlpha(
  const double goal_z,
  const double goal_t,
  const double start_t
);

/**
 * Initialize radial basis functions.
 *
 * \param widths widths of the RBFs, will be initialized
 * \param num_widths number of RBFs
 * \param centers centers of the RBFs, will be initialized
 * \param num_centers number of RBFs
 * \param goal_t time at the end of the DMP
 * \param start_t time at the start of the DMP
 * \param overlap value of each RBF at the center of the next RBF
 * \param alpha decay rate of the phase variable (default: 25.0 / 3.0)
 */
void initializeRbf(
  double* widths,
  int num_widths,
  double* centers,
  int num_centers,
  const double goal_t,
  const double start_t,
  const double overlap = 0.8,
  const double alpha = 8.33
);

/**
 * Represent trajectory as DMP.
 *
 * \note The final velocity will be calculated by numeric differentiation
 * from the data if allow_final_velocity is true. Otherwise we will assume
 * the final velocity to be zero. To reproduce the trajectory as closely as
 * possible, set the initial acceleration and velocity during execution to
 * zero, the final acceleration to zero and the final velocity to the value
 * that has been used during imitation.
 *
 * \param T time for each step of the trajectory
 * \param num_T number of steps
 * \param Y positions, contains num_T * num_dimensions entries in row-major
 *        order, i.e. the first position is located at the first num_dimensions
 *        entries of the array
 * \param num_steps number of steps
 * \param num_task_dims number of dimensions
 * \param weights weights that reproduce the trajectory (will be updated)
 * \param num_weights_per_dim number of features per dimension
 * \param num_weight_dims number of dimensions
 * \param widths widths of the radial basis functions (shared among DOFs)
 * \param num_widths number of RBFs
 * \param centers centers of the radial basis functions (shared among DOFs)
 * \param num_centers number of RBFs
 * \param regularization_coefficient can be set to solve instable problems
 *        where there are more weights that have to be learned than samples
 *        in the demonstrated trajectory (default: 1e-10)
 * \param alpha_y constant that has to be set for critical damping (default: 25)
 * \param beta_y constant that has to be set for critical damping (default: 25 / 4.0)
 * \param alpha_z decay rate of the phase variable (default: 25.0 / 3.0)
 * \param allow_final_velocity compute the final velocity from the data,
 *        otherwise we will assume it to be zero
 */
void imitate(
  const double* T,
  int num_T,
  const double* Y,
  int num_steps,
  int num_task_dims,
  double* weights,
  int num_weights_per_dim,
  int num_weight_dims,
  const double* widths,
  int num_widths,
  const double* centers,
  int num_centers,
  const double regularization_coefficient = 1e-10,
  const double alpha_y = 25.0,
  const double beta_y = 6.25,
  const double alpha_z = 8.33,
  bool allow_final_velocity = true
);

/**
 * Execute one step of the DMP.
 *
 * source: http://ijr.sagepub.com/content/32/3/263.full.pdf
 *
 * \param last_t time of last step (should equal t initially)
 * \param t current time
 * \param last_y last position
 * \param num_last_y number of dimensions
 * \param last_yd last velocity
 * \param num_last_yd number of dimensions
 * \param last_ydd last acceleration
 * \param num_last_ydd number of dimensions
 * \param y current position (will be updated)
 * \param num_y number of dimensions
 * \param yd velocity (will be updated)
 * \param num_yd number of dimensions
 * \param ydd acceleration (will be updated)
 * \param num_ydd number of dimensions
 * \param goal_y goal position
 * \param num_goal_y number of dimensions
 * \param goal_yd goal velocity
 * \param num_goal_yd number of dimensions
 * \param goal_ydd goal acceleration
 * \param num_goal_ydd number of dimensions
 * \param start_y start position
 * \param num_start_y number of dimensions
 * \param start_yd start velocity
 * \param num_start_yd number of dimensions
 * \param start_ydd start acceleration
 * \param num_start_ydd number of dimensions
 * \param goal_t time at the end of the DMP
 * \param start_t time at the start of the DMP
 * \param weights weights of the forcing term
 * \param num_weights_per_dim number of features per dimension
 * \param num_weight_dims number of dimensions
 * \param widths widths of the radial basis functions (shared among DOFs)
 * \param num_widths number of RBFs
 * \param centers centers of the radial basis functions (shared among DOFs)
 * \param num_centers number of RBFs
 * \param alpha_y constant that has to be set for critical damping (default: 25)
 * \param beta_y constant that has to be set for critical damping (default: 25 / 4.0)
 * \param alpha_z decay rate of the phase variable (default: 25.0 / 3.0)
 * \param integration_dt temporal step-size that will be used to integrate the
 *        velocity and position of the trajectory from the acceleration,
 *        smaller values will require more computation but will reproduce the
 *        demonstration more accurately
 */
void dmpStep(
  const double last_t,
  const double t,
  const double* last_y,
  int num_last_y,
  const double* last_yd,
  int num_last_yd,
  const double* last_ydd,
  int num_last_ydd,
  double* y,
  int num_y,
  double* yd,
  int num_yd,
  double* ydd,
  int num_ydd,
  const double* goal_y,
  int num_goal_y,
  const double* goal_yd,
  int num_goal_yd,
  const double* goal_ydd,
  int num_goal_ydd,
  const double* start_y,
  int num_start_y,
  const double* start_yd,
  int num_start_yd,
  const double* start_ydd,
  int num_start_ydd,
  const double goal_t,
  const double start_t,
  const double* weights,
  int num_weights_per_dim,
  int num_weight_dims,
  const double* widths,
  int num_widths,
  const double* centers,
  int num_centers,
  const double alpha_y = 25.0,
  const double beta_y = 6.25,
  const double alpha_z = 8.33,
  const double integration_dt = 0.001
);

/**
 * Represent trajectory as quaternion DMP.
 *
 * \note The final velocity will be calculated by numeric differentiation
 * from the data if allow_final_velocity is true. Otherwise we will assume
 * the final velocity to be zero. To reproduce the trajectory as closely as
 * possible, set the initial acceleration and velocity during execution to
 * zero, the final acceleration to zero and the final velocity to the value
 * that has been used during imitation.
 *
 * \param T time for each step of the trajectory
 * \param num_T number of steps
 * \param R rotations, contains num_T * 4 entries in row-major order, i.e.
 *        the first quaternion is located at the first 4 entries of the array
 * \param num_steps number of steps
 * \param num_task_dims should be 4
 * \param weights weights that reproduce the trajectory (will be updated)
 * \param num_weights_per_dim number of features per dimension
 * \param num_weight_dims should be 3
 * \param widths widths of the radial basis functions (shared among DOFs)
 * \param num_widths number of RBFs
 * \param centers centers of the radial basis functions (shared among DOFs)
 * \param num_centers number of RBFs
 * \param regularization_coefficient can be set to solve instable problems
 *        where there are more weights that have to be learned than samples
 *        in the demonstrated trajectory (default: 1e-10)
 * \param alpha_r constant that has to be set for critical damping (default: 25)
 * \param beta_r constant that has to be set for critical damping (default: 25 / 4.0)
 * \param alpha_z decay rate of the phase variable (default: 25.0 / 3.0)
 * \param allow_final_velocity compute the final velocity from the data,
 *        otherwise we will assume it to be zero
 */
void quaternionImitate(
  const double* T,
  int num_T,
  const double* R,
  int num_steps,
  int num_task_dims,
  double* weights,
  int num_weights_per_dim,
  int num_weight_dims,
  const double* widths,
  int num_widths,
  const double* centers,
  int num_centers,
  const double regularization_coefficient = 1e-10,
  const double alpha_r = 25.0,
  const double beta_r = 6.25,
  const double alpha_z = 8.33,
  bool allow_final_velocity = true
);

/**
 * Execute one step of the Quaternion DMP.
 *
 * source: http://ieeexplore.ieee.org/document/6907291/?arnumber=6907291
 *
 * \param last_t time of last step (should equal t initially)
 * \param t current time
 * \param last_r last rotation
 * \param num_last_r should be 4
 * \param last_rd last rotational velocity
 * \param num_last_rd should be 3
 * \param last_rdd last rotational acceleration
 * \param num_last_rdd should be 3
 * \param r current rotation (will be updated)
 * \param num_r should be 4
 * \param rd rotational velocity (will be updated)
 * \param num_rd should be 3
 * \param rdd rotational acceleration (will be updated)
 * \param num_rdd should be 3
 * \param goal_r final rotation
 * \param num_goal_r should be 4
 * \param goal_rd final rotational velocity
 * \param num_goal_rd should be 3
 * \param goal_rdd final rotational acceleration
 * \param num_goal_rdd should be 3
 * \param start_r first rotation
 * \param num_start_r should be 4
 * \param start_rd first rotational velocity
 * \param num_start_rd should be 3
 * \param start_rdd first rotational acceleration
 * \param num_start_rdd should be 3
 * \param goal_t time at the end of the DMP
 * \param start_t time at the start of the DMP
 * \param weights weights of the forcing term
 * \param num_weights_per_dim number of features per dimension
 * \param num_weight_dims should be 3
 * \param widths widths of the radial basis functions (shared among DOFs)
 * \param num_widths number of RBFs
 * \param centers centers of the radial basis functions (shared among DOFs)
 * \param num_centers number of RBFs
 * \param alpha_r constant that has to be set for critical damping (default: 25)
 * \param beta_r constant that has to be set for critical damping (default: 25 / 4.0)
 * \param alpha_z decay rate of the phase variable (default: 25.0 / 3.0)
 * \param integration_dt temporal step-size that will be used to integrate the
 *        velocity and position of the trajectory from the acceleration,
 *        smaller values will require more computation but will reproduce the
 *        demonstration more accurately
 */
void quaternionDmpStep(
  const double last_t,
  const double t,
  const double* last_r,
  int num_last_r,
  const double* last_rd,
  int num_last_rd,
  const double* last_rdd,
  int num_last_rdd,
  double* r,
  int num_r,
  double* rd,
  int num_rd,
  double* rdd,
  int num_rdd,
  const double* goal_r,
  int num_goal_r,
  const double* goal_rd,
  int num_goal_rd,
  const double* goal_rdd,
  int num_goal_rdd,
  const double* start_r,
  int num_start_r,
  const double* start_rd,
  int num_start_rd,
  const double* start_rdd,
  int num_start_rdd,
  const double goal_t,
  const double start_t,
  const double* weights,
  int num_weights_per_dim,
  int num_weight_dims,
  const double* widths,
  int num_widths,
  const double* centers,
  int num_centers,
  const double alpha_r,
  const double beta_r,
  const double alpha_z,
  const double integration_dt
);

namespace internal
{

void compute_gradient(
  const double* in,
  int num_in_steps,
  int num_in_dims,
  double* out,
  int num_out_steps,
  int num_out_dims,
  const double* time,
  int num_time,
  bool allow_final_velocity = true
);

void compute_quaternion_gradient(
  const double* in,
  int num_in_steps,
  int num_in_dims,
  double* out,
  int num_out_steps,
  int num_out_dims,
  const double* time,
  int num_time,
  bool allow_final_velocity = true
);

} // namespace internal

} // namespace Dmp
