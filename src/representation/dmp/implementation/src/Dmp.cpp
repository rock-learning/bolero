#include <Dmp.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <vector>
#include <cassert>
#include <stdexcept>


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
 * Calculates the gradient function for \p in, e.g. the derivation.
 * The returned gradient has the same shape as the input array.
 */
template <typename PosType, typename VelType, typename TimeType>
void gradient(
  const PosType& in,
  VelType& out,
  const TimeType& time,
  const bool allow_final_velocity
)
{
  assert(in.cols() > 1);
  out.resize(in.rows(), in.cols());

  // Special case for first element: assume gradient to be zero
  out.col(0).setZero();

  const int end = in.cols();
  for(int i = 1; i < end; ++i)
  {// Difference quotient for each following element
    out.col(i) = (in.col(i) - in.col(i - 1)) / (time(i) - time(i - 1));
  }
  if(!allow_final_velocity)
    out.col(end - 1).setZero();
}

/**
 * Compute axis-angle representation from quaternion (logarithmic map).
 */
Eigen::Array3d qLog(const Eigen::Quaterniond& q);


/**
 * Compute quaternion from axis-angle representation (exponential map).
 */
Eigen::Quaterniond vecExp(const Eigen::Vector3d& input);


typedef std::vector<Eigen::Quaternion<double>, Eigen::aligned_allocator<Eigen::Quaternion<double> > > QuaternionVector;


template <typename VelType, typename TimeType>
void quaternionGradient(
  const QuaternionVector& rotations,
  VelType& velocities,
  const TimeType& time,
  bool allow_final_velocity)
{
  assert(velocities.rows() == 3);
  assert((size_t) velocities.cols() == rotations.size());
  assert(rotations.size() >= 2);

  // Special case for first element: assume gradient to be zero
  velocities.col(0).setZero();

  // Backward difference quotient
  const int end = (int) rotations.size();
  for(int i = 1; i < end; ++i)
  {// Difference quotient for each following element
    const Eigen::Quaterniond& q0 = rotations[i - 1];
    const Eigen::Quaterniond& q1 = rotations[i];
    const double dt = time(i) - time(i - 1);
    velocities.col(i) = 2 * qLog(q1 * q0.conjugate()) / dt;
  }
  if(!allow_final_velocity)
    velocities.col(end - 1).setZero();
}


/**
 * Solve for 6 position, velocity, and acceleration constraints.
 */
void solveConstraints(
  const double t0,
  const double t1,
  const Eigen::ArrayXd y0,
  const Eigen::ArrayXd y0d,
  const Eigen::ArrayXd y0dd,
  const Eigen::ArrayXd y1,
  const Eigen::ArrayXd y1d,
  const Eigen::ArrayXd y1dd,
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1> > >& coefficients
);


const Eigen::MatrixXd rbfDesignMatrix(
  const Eigen::ArrayXd& T,
  const double alpha_z,
  const Eigen::ArrayXd& widths,
  const Eigen::ArrayXd& centers
);


/**
 * Linear regression with L2 regularization.
 *
 * \param X design matrix, each column contains a sample
 * \param targets each column contains a sample
 * \param regularization_coefficient
 * \param weights resulting weights (will be updated)
 */
void ridgeRegression(
  const Eigen::MatrixXd& X,
  const Eigen::ArrayXXd& targets,
  const double regularization_coefficient,
  Eigen::Map<Eigen::ArrayXXd>& weights
);


/**
 * Apply 6 position, velocity, and acceleration constraints.
 */
void applyConstraints(
  const double t, const Eigen::ArrayXd& goal_y, const double goal_t,
  const std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1> > >& coefficients,
  Eigen::ArrayXd& g, Eigen::ArrayXd& gd, Eigen::ArrayXd& gdd
);


/**
 * Determine accelerating forces of the forcing term during the demonstrated
 * trajectory.
 */
void determineForces(
  const Eigen::ArrayXd& T,
  const Eigen::ArrayXXd& Y,
  Eigen::ArrayXXd& F,
  const double alpha_y,
  const double beta_y,
  bool allow_final_velocity
);


const Eigen::ArrayXd rbfActivations(
  const double z,
  const Eigen::ArrayXd& widths,
  const Eigen::ArrayXd& centers,
  const bool normalized = true
);


const Eigen::ArrayXd forcingTerm(
  const double z,
  const Eigen::ArrayXXd& weights,
  const Eigen::ArrayXd& widths,
  const Eigen::ArrayXd& centers
);


double calculateAlpha(
  const double goal_z,
  const double goal_t,
  const double start_t
)
{
  if(goal_z <= 0.0)
    throw std::invalid_argument("Final phase must be > 0!");
  if(start_t >= goal_t)
    throw std::invalid_argument("Goal must be chronologically after start!");

  const double int_dt = 0.001;
  const double execution_time = goal_t - start_t;
  const int num_phases = (int)(execution_time / int_dt) + 1;
  // assert that the execution_time is approximately divisible by int_dt
  assert(abs(((num_phases -1) * int_dt) - execution_time) < 0.05);
  return (1.0 - pow(goal_z, 1.0 / (num_phases - 1))) * (num_phases - 1);
}

void initializeRbf(
  double* widths,
  int num_widths,
  double* centers,
  int num_centers,
  const double goal_t,
  const double start_t,
  const double overlap,
  const double alpha
)
{
  const int num_weights_per_dim = num_widths;
  if(num_widths <= 1)
    throw std::invalid_argument("The number of weights per dimension must be > 1!");

  if(start_t >= goal_t)
    throw std::invalid_argument("Goal must be chronologically after start!");
  const double execution_time = goal_t - start_t;

  assert(num_weights_per_dim == num_widths);
  assert(num_weights_per_dim == num_centers);
  Eigen::Map<Eigen::ArrayXd> widths_array(widths, num_widths);
  Eigen::Map<Eigen::ArrayXd> centers_array(centers, num_centers);

  const double step = execution_time / (num_weights_per_dim - 1); // -1 because we want the last entry to be execution_time
  const double logOverlap = -std::log(overlap);
  // do first iteration outside loop because we need access to i and i - 1 in loop
  double t = start_t;
  centers_array(0) = phase(t, alpha, goal_t, start_t);
  for(int i = 1; i < num_weights_per_dim; ++i)
  {
    // Alternatively Eigen::LinSpaced can be used, however it does exactly the same calculation
    t = i * step; // normally lower_border + i * step but lower_border is 0
    centers_array(i) = phase(t, alpha, goal_t, start_t);
    // Choose width of RBF basis functions automatically so that the
    // RBF centered at one center has value overlap at the next center
    const double diff = centers_array(i) - centers_array(i - 1);
    widths_array(i - 1) = logOverlap / (diff * diff);
  }
  // Width of last gausian cannot be calculated, just use the same width as the one before
  widths_array(num_weights_per_dim - 1) = widths_array(num_weights_per_dim - 2);
}


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
  const double regularization_coefficient,
  const double alpha_y,
  const double beta_y,
  const double alpha_z,
  bool allow_final_velocity
)
{
  assert(num_steps == num_T);
  assert(num_weights_per_dim == num_widths);
  assert(num_weights_per_dim == num_centers);
  assert(num_task_dims == num_weight_dims);

  if(regularization_coefficient < 0.0)
  {
    throw std::invalid_argument("Regularization coefficient must be >= 0!");
  }

  Eigen::Map<const Eigen::ArrayXd> widths_array(widths, num_widths);
  Eigen::Map<const Eigen::ArrayXd> centers_array(centers, num_centers);

  Eigen::ArrayXd T_array = Eigen::Map<const Eigen::ArrayXd>(T, num_T);
  Eigen::ArrayXXd Y_array = Eigen::Map<const Eigen::ArrayXXd>(Y, num_task_dims, num_steps);
  Eigen::ArrayXXd F(num_task_dims, num_steps);
  determineForces(T_array, Y_array, F, alpha_y, beta_y, allow_final_velocity);

  const Eigen::MatrixXd X = rbfDesignMatrix(T_array, alpha_z, widths_array, centers_array);

  Eigen::Map<Eigen::ArrayXXd> weights_array(weights, num_task_dims, num_weights_per_dim);
  ridgeRegression(X, F, regularization_coefficient, weights_array);
}

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
  const double alpha_y,
  const double beta_y,
  const double alpha_z,
  const double integration_dt
)
{
  const int num_dimensions = num_last_y;
  if(start_t >= goal_t)
    throw std::invalid_argument("Goal must be chronologically after start!");

  assert(num_dimensions == num_last_y);
  assert(num_dimensions == num_last_yd);
  assert(num_dimensions == num_last_ydd);
  Eigen::Map<const Eigen::ArrayXd> last_y_array(last_y, num_last_y);
  Eigen::Map<const Eigen::ArrayXd> last_yd_array(last_yd, num_last_yd);
  Eigen::Map<const Eigen::ArrayXd> last_ydd_array(last_ydd, num_last_ydd);

  assert(num_dimensions == num_y);
  assert(num_dimensions == num_yd);
  assert(num_dimensions == num_ydd);
  Eigen::Map<Eigen::ArrayXd> y_array(y, num_y);
  Eigen::Map<Eigen::ArrayXd> yd_array(yd, num_yd);
  Eigen::Map<Eigen::ArrayXd> ydd_array(ydd, num_ydd);

  assert(num_dimensions == num_goal_y);
  assert(num_dimensions == num_goal_yd);
  assert(num_dimensions == num_goal_ydd);
  Eigen::Map<const Eigen::ArrayXd> goal_y_array(goal_y, num_goal_y);
  Eigen::Map<const Eigen::ArrayXd> goal_yd_array(goal_yd, num_goal_yd);
  Eigen::Map<const Eigen::ArrayXd> goal_ydd_array(goal_ydd, num_goal_ydd);

  assert(num_dimensions == num_start_y);
  assert(num_dimensions == num_start_yd);
  assert(num_dimensions == num_start_ydd);
  Eigen::Map<const Eigen::ArrayXd> start_y_array(start_y, num_start_y);
  Eigen::Map<const Eigen::ArrayXd> start_yd_array(start_yd, num_start_yd);
  Eigen::Map<const Eigen::ArrayXd> start_ydd_array(start_ydd, num_start_ydd);

  if(t <= start_t)
  {
    y_array = start_y_array;
    yd_array = start_yd_array;
    ydd_array = start_ydd_array;
  }
  else
  {
    const double execution_time = goal_t - start_t;

    // TODO only recompute iff start or goal state changes
    std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1> > > coefficients;
    solveConstraints(
        start_t, goal_t,
        start_y_array, start_yd_array, start_ydd_array,
        goal_y_array, goal_yd_array, goal_ydd_array,
        coefficients);

    y_array = last_y_array;
    yd_array = last_yd_array;
    ydd_array = last_ydd_array;
    Eigen::ArrayXd g(num_y);
    Eigen::ArrayXd gd(num_y);
    Eigen::ArrayXd gdd(num_y);

    assert(num_weights_per_dim == num_widths);
    assert(num_weights_per_dim == num_centers);
    assert(num_weight_dims == num_dimensions);
    Eigen::Map<const Eigen::ArrayXXd> weights_array(
        weights, num_dimensions, num_weights_per_dim);
    Eigen::Map<const Eigen::ArrayXd> widths_array(widths, num_widths);
    Eigen::Map<const Eigen::ArrayXd> centers_array(centers, num_centers);

    // We use multiple integration steps to improve numerical precision
    double current_t = last_t;
    while(current_t < t)
    {
      double dt_int = integration_dt;
      if(t - current_t < dt_int)
        dt_int = t - current_t;

      current_t += dt_int;

      const double z = phase(current_t, alpha_z, goal_t, start_t);
      const Eigen::ArrayXd f = forcingTerm(z, weights_array, widths_array, centers_array);

      applyConstraints(current_t, goal_y_array, goal_t, coefficients, g, gd, gdd);
      const double execution_time_squared = execution_time * execution_time;
      ydd_array = (alpha_y
                   * (beta_y * (g - y_array)
                      + execution_time * gd
                      - execution_time * yd_array)
                   + gdd * execution_time_squared + f)
                  / execution_time_squared;
      y_array += dt_int * yd_array;
      yd_array += dt_int * ydd_array;
    }
    assert(current_t == t);
  }
}


const double phase(
  const double t,
  const double alpha,
  const double goal_t,
  const double start_t
)
{
  const double int_dt = 0.001;
  const double execution_time = goal_t - start_t;
  const double b = std::max(1 - alpha * int_dt / execution_time, 1e-10);
  return pow(b, (t - start_t) / int_dt);
}


void solveConstraints(
  const double t0,
  const double t1,
  const Eigen::ArrayXd y0,
  const Eigen::ArrayXd y0d,
  const Eigen::ArrayXd y0dd,
  const Eigen::ArrayXd y1,
  const Eigen::ArrayXd y1d,
  const Eigen::ArrayXd y1dd,
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1> > >& coefficients
)
{
  const double t02 = t0 * t0;
  const double t03 = t02 * t0;
  const double t04 = t03 * t0;
  const double t05 = t04 * t0;
  const double t12 = t1 * t1;
  const double t13 = t12 * t1;
  const double t14 = t13 * t1;
  const double t15 = t14 * t1;

  Eigen::Matrix<double, 6, 6> M;
  M << 1,   t0,      t02,      t03,       t04,        t05,
       0,   1,   2 * t0,   3 * t02,   4 * t03,    5 * t04,
       0,   0,       2,    6 * t0,   12 * t02,   20 * t03,
       1,   t1,      t12,      t13,       t14,        t15,
       0,   1,   2 * t1,   3 * t12,   4 * t13,    5 * t14,
       0,   0,       2,    6 * t1,   12 * t12,   20 * t13;

  // Solve M*b = y for b in each DOF separately
  Eigen::PartialPivLU<Eigen::Matrix<double, 6, 6> > luOfM(M);
  coefficients.clear();
  coefficients.reserve(y0.size());
  Eigen::Matrix<double, 6, 1> x;
  for(unsigned i = 0; i < y0.size(); ++i)
  {
    x << y0[i], y0d[i], y0dd[i], y1[i], y1d[i], y1dd[i];
    coefficients.push_back(luOfM.solve(x));
  }
}


void applyConstraints(
  const double t, const Eigen::ArrayXd& goal_y, const double goal_t,
  const std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1> > >& coefficients,
  Eigen::ArrayXd& g, Eigen::ArrayXd& gd, Eigen::ArrayXd& gdd
)
{
  if(t > goal_t)
  {
    /**For t > goal_t the polynomial should always 'pull' to the goal position.
     * But velocity and acceleration should be zero.
     * This is done to avoid diverging from the goal if the dmp is executed
     * longer than expected. */
    g = goal_y;
    gd.setZero();
    gdd.setZero();
  }
  else
  {
    Eigen::Matrix<double, 1, 6> pos;
    Eigen::Matrix<double, 1, 6> vel;
    Eigen::Matrix<double, 1, 6> acc;
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double t4 = t3 * t;
    const double t5 = t4 * t;
    pos << 1,   t,      t2,       t3,        t4,        t5;
    vel << 0,   1,   2 * t,   3 * t2,    4 * t3,    5 * t4;
    acc << 0,   0,       2,    6 * t,   12 * t2,   20 * t3;

    for(int i = 0; i < g.size(); ++i)
    {
      g[i] = pos * coefficients[i];
      gd[i] = vel * coefficients[i];
      gdd[i] = acc * coefficients[i];
    }
  }
}


void determineForces(
  const Eigen::ArrayXd& T,
  const Eigen::ArrayXXd& Y,
  Eigen::ArrayXXd& F,
  const double alpha_y,
  const double beta_y,
  bool allow_final_velocity
)
{
  assert(T.rows() == Y.cols());
  assert(T.rows() == F.cols());
  assert(Y.rows() == F.rows());
  const int num_dim = Y.rows();
  const int num_steps = Y.cols();

  Eigen::ArrayXXd Yd(num_dim, num_steps);
  gradient(Y, Yd, T, allow_final_velocity);
  Eigen::ArrayXXd Ydd(num_dim, num_steps);
  // The final acceleration is needs to be zero for the imitation learning to
  // work properly in Muelling DMPs.
  gradient(Yd, Ydd, T, false);

  //following code is equation (9) from [Muelling2012]
  Eigen::VectorXd start_y(Y.col(0));
  Eigen::VectorXd start_yd(Yd.col(0));
  Eigen::VectorXd start_ydd(Ydd.col(0));
  Eigen::VectorXd goal_y(Y.col(num_steps - 1));
  Eigen::VectorXd goal_yd(Yd.col(num_steps - 1));
  Eigen::VectorXd goal_ydd(Ydd.col(num_steps - 1));
  const double start_t = T(0);
  const double goal_t = T(num_steps - 1);

  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1> > > coefficients;
  solveConstraints(start_t, goal_t, start_y, start_yd, start_ydd,
                   goal_y, goal_yd, goal_ydd, coefficients);

  Eigen::ArrayXd g(num_dim);
  Eigen::ArrayXd gd(num_dim);
  Eigen::ArrayXd gdd(num_dim);
  const double t = goal_t - start_t;
  const double t2 = t * t;

  for(int i = 0; i < num_steps; ++i)
  {
    applyConstraints(T(i), goal_y, goal_t, coefficients, g, gd, gdd);
    F.col(i) = t2 * Ydd.col(i)
               - alpha_y * (beta_y * (g - Y.col(i))
                            + gd * t
                            - Yd.col(i) * t)
               - t2 * gdd;
  }
}


const Eigen::MatrixXd rbfDesignMatrix(
  const Eigen::ArrayXd& T,
  const double alpha_z,
  const Eigen::ArrayXd& widths,
  const Eigen::ArrayXd& centers
)
{
  Eigen::MatrixXd X(centers.rows(), T.rows());
  for(int i = 0; i < T.rows(); i++)
  {
    const double z = phase(T(i), alpha_z, T(T.rows() - 1), T(0));
    X.col(i) = z * rbfActivations(z, widths, centers, true);
  }
  return X;
}


// pseudo inverse from http://eigen.tuxfamily.org/bz/show_bug.cgi?id=257
Eigen::MatrixXd pseudoInverse(const Eigen::MatrixXd& a, double epsilon = std::numeric_limits<double>::epsilon())
{
  Eigen::JacobiSVD<Eigen::MatrixXd> svd = a.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

  double tolerance = epsilon * std::max(a.cols(), a.rows()) * svd.singularValues().array().abs().maxCoeff();

  return svd.matrixV() * Eigen::MatrixXd( (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().
      array().inverse(), 0) ).asDiagonal() * svd.matrixU().adjoint();
}


void ridgeRegression(
  const Eigen::MatrixXd& X,
  const Eigen::ArrayXXd& targets,
  const double regularization_coefficient,
  Eigen::Map<Eigen::ArrayXXd>& weights
)
{
  const int num_outputs = weights.rows();
  const int num_features = weights.cols();
  for(int i = 0; i < num_outputs; i++)
    weights.row(i) =
        (pseudoInverse(X * X.transpose()
          + regularization_coefficient * Eigen::MatrixXd::Identity(num_features, num_features)
          )
         * X * targets.row(i).transpose().matrix()
         ).transpose();
}


const Eigen::ArrayXd rbfActivations(
  const double z,
  const Eigen::ArrayXd& widths,
  const Eigen::ArrayXd& centers,
  const bool normalized
)
{
  Eigen::ArrayXd activations = (-widths * (z - centers).pow(2)).exp();
  if(normalized)
    activations /= activations.sum();
  return activations;
}


const Eigen::ArrayXd forcingTerm(
  const double z,
  const Eigen::ArrayXXd& weights,
  const Eigen::ArrayXd& widths,
  const Eigen::ArrayXd& centers
)
{
  const Eigen::ArrayXd activations = rbfActivations(z, widths, centers, true);
  return (z * weights.matrix() * activations.matrix()).array();
}


Eigen::Array3d qLog(const Eigen::Quaterniond& q)
{
  const double len = q.vec().norm();
  if(len == 0.0)
    return Eigen::Array3d::Zero();
  return q.vec().array() / len * acos(q.w());
}


Eigen::Quaterniond vecExp(const Eigen::Vector3d& input)
{
  const double len = input.norm();
  if(len != 0)
  {
    const Eigen::Array3d vec = sin(len) * input / len;
    return Eigen::Quaterniond(cos(len), vec.x(), vec.y(), vec.z());
  }
  else
  {
    return Eigen::Quaterniond::Identity();
  }
}


void quaternionDetermineForces(
  const Eigen::ArrayXd& T,
  const QuaternionVector& R,
  Eigen::ArrayXXd& F,
  const double alpha_y,
  const double beta_y,
  bool allow_final_velocity
)
{
  assert((size_t) T.rows() == R.size());
  assert(T.rows() == F.cols());
  assert((size_t) F.cols() == R.size());
  const int num_steps = T.rows();

  Eigen::ArrayXXd Rd(3, num_steps);
  quaternionGradient(R, Rd, T, allow_final_velocity);
  Eigen::ArrayXXd Rdd(3, num_steps);
  gradient(Rd, Rdd, T, false);

  const double t = T(num_steps - 1) - T(0);
  const double t2 = t * t;

  // Following code is equation (16) from [Ude2014] rearranged to $f_0$
  for(size_t i = 0; i < R.size(); ++i)
  {
    F.col(i) = t2 * Rdd.col(i)
               - (alpha_y * (beta_y * 2 * qLog(R.back() * R[i].conjugate())
                             - t * Rd.col(i)));
  }
}


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
  const double regularization_coefficient,
  const double alpha_r,
  const double beta_r,
  const double alpha_z,
  bool allow_final_velocity
)
{
  assert(num_steps == num_T);
  assert(num_task_dims == 4);
  assert(num_weights_per_dim == num_widths);
  assert(num_weights_per_dim == num_centers);
  assert(num_weight_dims == 3);

  if(regularization_coefficient < 0.0)
  {
    throw std::invalid_argument("Regularization coefficient must be >= 0!");
  }

  Eigen::Map<const Eigen::ArrayXd> widths_array(widths, num_widths);
  Eigen::Map<const Eigen::ArrayXd> centers_array(centers, num_centers);

  Eigen::ArrayXd T_array = Eigen::Map<const Eigen::ArrayXd>(T, num_T);
  Eigen::ArrayXXd R_array = Eigen::Map<const Eigen::ArrayXXd>(R, num_task_dims, num_steps);
  QuaternionVector rotationsVector;
  for(int i = 0; i < num_steps; ++i)
  {
    Eigen::Quaterniond q(R_array(0, i), R_array(1, i), R_array(2, i), R_array(3, i));
    q.normalize(); // has to be done to avoid nans
    rotationsVector.push_back(q);
  }
  Eigen::ArrayXXd F(num_weight_dims, num_steps);
  quaternionDetermineForces(T_array, rotationsVector, F, alpha_r, beta_r, allow_final_velocity);

  const Eigen::MatrixXd X = rbfDesignMatrix(T_array, alpha_z, widths_array, centers_array);

  Eigen::Map<Eigen::ArrayXXd> weights_array(weights, num_weight_dims, num_weights_per_dim);
  ridgeRegression(X, F, regularization_coefficient, weights_array);
}

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
)
{
  if(start_t >= goal_t)
    throw std::invalid_argument("Goal must be chronologically after start!");

  assert(4 == num_last_r);
  assert(3 == num_last_rd);
  assert(3 == num_last_rdd);
  const Eigen::Quaterniond last_r_array(last_r[0], last_r[1], last_r[2], last_r[3]);
  Eigen::Map<const Eigen::ArrayXd> last_rd_array(last_rd, num_last_rd);
  Eigen::Map<const Eigen::ArrayXd> last_rdd_array(last_rdd, num_last_rdd);

  assert(4 == num_r);
  assert(3 == num_rd);
  assert(3 == num_rdd);
  Eigen::Quaterniond r_array(r[0], r[1], r[2], r[3]);
  Eigen::Map<Eigen::ArrayXd> rd_array(rd, num_rd);
  Eigen::Map<Eigen::ArrayXd> rdd_array(rdd, num_rdd);

  assert(4 == num_goal_r);
  assert(3 == num_goal_rd);
  assert(3 == num_goal_rdd);
  const Eigen::Quaterniond goal_r_array(goal_r[0], goal_r[1], goal_r[2], goal_r[3]);
  Eigen::Map<const Eigen::ArrayXd> goal_rd_array(goal_rd, num_goal_rd);
  Eigen::Map<const Eigen::ArrayXd> goal_rdd_array(goal_rdd, num_goal_rdd);

  assert(4 == num_start_r);
  assert(3 == num_start_rd);
  assert(3 == num_start_rdd);
  const Eigen::Quaterniond start_r_array(start_r[0], start_r[1], start_r[2], start_r[3]);
  Eigen::Map<const Eigen::ArrayXd> start_rd_array(start_rd, num_start_rd);
  Eigen::Map<const Eigen::ArrayXd> start_rdd_array(start_rdd, num_start_rdd);

  if(t <= start_t)
  {
    r_array = start_r_array;
    rd_array = start_rd_array;
    rdd_array = start_rdd_array;
  }
  else
  {
    const double execution_time = goal_t - start_t;

    r_array = last_r_array;
    rd_array = last_rd_array;
    rdd_array = last_rdd_array;

    assert(num_weights_per_dim == num_widths);
    assert(num_weights_per_dim == num_centers);
    assert(num_weight_dims == 3);
    Eigen::Map<const Eigen::ArrayXXd> weights_array(
        weights, num_weight_dims, num_weights_per_dim);
    Eigen::Map<const Eigen::ArrayXd> widths_array(widths, num_widths);
    Eigen::Map<const Eigen::ArrayXd> centers_array(centers, num_centers);

    // We use multiple integration steps to improve numerical precision
    double current_t = last_t;
    while(current_t < t)
    {
      double dt_int = integration_dt;
      if(t - current_t < dt_int)
        dt_int = t - current_t;

      current_t += dt_int;

      const double z = phase(current_t, alpha_z, goal_t, start_t);
      const Eigen::ArrayXd f = forcingTerm(z, weights_array, widths_array, centers_array);

      const double execution_time_squared = execution_time * execution_time;

      rdd_array = (alpha_r * (beta_r * 2.0 * qLog(goal_r_array * r_array.conjugate())
                              - execution_time * rd_array)
                   + f)
                  / execution_time_squared;
      r_array = vecExp(dt_int / 2.0 * rd_array) * r_array;
      rd_array += dt_int * rdd_array;
    }
    assert(current_t == t);
  }

  r[0] = r_array.w();
  r[1] = r_array.x();
  r[2] = r_array.y();
  r[3] = r_array.z();
}

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
  bool allow_final_velocity
)
{
  assert(num_in_steps == num_time);
  assert(num_out_steps == num_time);
  assert(num_in_dims == num_out_dims);
  Eigen::Map<const Eigen::ArrayXXd> in_array(in, num_in_dims, num_time);
  Eigen::Map<Eigen::ArrayXXd> out_array(out, num_out_dims, num_time);
  Eigen::Map<const Eigen::ArrayXd> time_array(time, num_time);
  gradient(in_array, out_array, time_array, allow_final_velocity);
}


void compute_quaternion_gradient(
  const double* in,
  int num_in_steps,
  int num_in_dims,
  double* out,
  int num_out_steps,
  int num_out_dims,
  const double* time,
  int num_time,
  bool allow_final_velocity
)
{
  assert(num_in_steps == num_time);
  assert(num_out_steps == num_time);
  assert(num_in_dims == 4);
  assert(num_out_dims == 3);

  Eigen::Map<const Eigen::ArrayXXd> in_array(in, num_in_dims, num_time);
  QuaternionVector rotationsVector;
  for(int i = 0; i < num_time; ++i)
  {
    Eigen::Quaterniond q(in_array(0, i), in_array(1, i), in_array(2, i), in_array(3, i));
    q.normalize(); // has to be done to avoid nans
    rotationsVector.push_back(q);
  }
  Eigen::Map<Eigen::ArrayXXd> out_array(out, num_out_dims, num_time);
  Eigen::Map<const Eigen::ArrayXd> time_array(time, num_time);
  quaternionGradient(rotationsVector, out_array, time_array, allow_final_velocity);
}

} // namespace internal

} // namespace Dmp
