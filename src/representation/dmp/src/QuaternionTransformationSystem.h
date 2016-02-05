#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <vector>

namespace dmp
{
class ForcingTerm;
/**
 * A transformation system that works with quaternions.
 * Based on:
 *   [Ude2014] Ude, Ales; Nemec Bojan; Petric, Tadej; Morimoto, Jun;
 *             Orientation in Cartesian Space Dynamic Movement Primitives,
 *             <http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6907291>`_
 *             2014 IEEE Internation Conference on Robotics and Automation (ICRA), 2014.
 *
 * @note You should have read and understood the normal Dmp (Dmp.h/cpp) before trying
 *       to understand the quaternion dmp.
 *
*/
class QuaternionTransformationSystem
{
public:
  typedef std::vector<Eigen::Quaternion<double>, Eigen::aligned_allocator<Eigen::Quaternion<double> > > QuaternionVector;


  QuaternionTransformationSystem(ForcingTerm& ft, const double executionTime,
                                 const double dt, const double alphaZ = 25,
                                 const double betaZ = 6.25);

  /** Initializes the transformation system.
   *  The system can be re-initialized at any time to change the end position.
   * @note For the quaternion transformation system the endVelocity and endAcceleration
   *       is always [0, 0, 0].
   * @param startPos Starting position of the dmp
   * @param startVel starting velocity in angle-axis notation. I.e. [x, y, z] is
   *                 the axis of rotation and the length of the vector is the
   *                 amount of rotation in radians.
   */
  void initialize(const Eigen::Quaterniond& startPos, const Eigen::Array3d& startVel,
                  const Eigen::Quaterniond& endPos);

  /**
  * Calculates the next rotation based on the current rotation and the current
  * phase.
  * @param[in] phase The current phase as calculated by the canonical system.
  * @param[in,out] The current position, will be updated with the new position.
  */
  void executeStep(const double phase, Eigen::Quaterniond& position);

  /**Determine the required forces to generate a demonstrated trajectory.
  * \param[in] rotations The rotations of the trajectory.
  * \param[in, out] velocities The velocities in AngleAxis notation. Each column should be
  *                            one velocity. Will be approximated if empty.
  * \param[in, out] accelerations The accelerations in AngleAxis notation. Each column
  *                               should contain one acceleration. Will be
  *                               approximated if empty.
  */
  static void determineForces(const QuaternionVector& rotations,
                       Eigen::ArrayXXd& velocities,
                       Eigen::ArrayXXd& accelerations,
                       Eigen::ArrayXXd& forces,
                       const double dt, const double executionTime,
                       const double alphaZ = 25.0, const double betaZ = 6.25);

private:

  /**
  * Calculates the gradient of a vector of quaternions.
  * uses forward difference quotient for the first element, backward difference for the last
  * and central difference quotient for all other elements.
  * @param[in] rotations A vector of quaternions that should be derived
  * @param[out] velocities A vector of angular velocities encoded as angle axis
  * @param[in] dt dt...
  *
  * @note Use this method to approximate the first derivative the
  */
  static void gradient(const QuaternionVector& rotations,
                       Eigen::ArrayXXd& velocities, const double dt);


  /**
  * Calculates the exponential of a vector based on equation
  * (25) from [Ude2014]
  */
  Eigen::Quaterniond vecExp(const Eigen::Vector3d& vec) const;

  /**
  * Calculates the logarithm of the specified unit quaternion based on
  * equation (19) from [Ude2014]
  */
  static Eigen::Array3d qLog(const Eigen::Quaterniond& q);

  const double executionTime;
  const double dt;
  const double alphaZ;
  const double betaZ;
  ForcingTerm& forcingTerm;

  Eigen::Quaterniond startPos;
  Eigen::Quaterniond endPos;
  Eigen::Array3d eta; //**<This is the current state of the dynamical system. velocity = (1/2 eta * q) / T */
  bool initialized;
};
}
