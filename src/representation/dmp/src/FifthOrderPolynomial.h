/*
 * @file FifthOrderPolynomial.h
 * @author Arne Boeckmann (arne.boeckmann@dfki.de)
 */

#pragma once
#include <Eigen/Core>
#include <Eigen/LU>
#include<Eigen/StdVector>
#include "CanonicalSystem.h"
#include <vector>

namespace dmp {

/** Fifth order polynomial that models the movement of the goal.
 * The polynomial is used to model the movement of the goal in [Muelling2012]
 *  and [Muelling2011].
 *
 *  @param CS Type of the CanonicalSystem
 */
class FifthOrderPolynomial {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**Creates an invalid instance of the polynomial.
   * setConstraints() needs to be called before the instance can be used*/
  FifthOrderPolynomial();

  /**
   * Set constraints of the fifth order polynomial.
   * @param gt Start position of the trajectory
   * @param gdt Start velocity
   * @param gddt Start acceleration
   * @param gf Goal position of the trajectory
   * @param gdf Goal velocity
   * @param gddf Goal acceleration
   * @param startTime time at which the polynomial should be at gt, gdt, gddt
   * @param endTime time at which the polynomial should be at gf, gdf, gddf
   *
   * @note gt, gdt, gddt, gf, gdf and gddf should have the same size.
   *       This size should be the same as the number of task space dimensions.
   * @note startTime should be < endTime
   */
  void setConstraints(const Eigen::VectorXd& gt, const Eigen::VectorXd& gdt,
                      const Eigen::VectorXd& gddt, const Eigen::VectorXd& gf,
                      const Eigen::VectorXd& gdf, const Eigen::VectorXd& gddf,
                      const double startTime, const double endTime);

  /**
   * Returns the position, velocity and acceleration
   * of the fifth order polynomial at the given time.
   *
   * If t is > endTime the goal position will be returned
   *
   * @note outPosition, outVelocity and outAcceleration should have the same size.
   *       Their size should be equal to the number of task space dimensions.
   *       If their size is 0, nothing will be calculated.
   */
  void getValueAt(const double t, Eigen::ArrayXd& outPosition, Eigen::ArrayXd& outvelocity,
      Eigen::ArrayXd& outAcceleration) const;

private:
  typedef Eigen::Matrix<double, 6, 6> Mtype;
  typedef Eigen::Matrix<double, 6, 1> Vector6d;

  /** Contains a vector for each DOF.
   *  This vector contains the coefficients that define the shape
   *  of the polynomial. */
  std::vector<Vector6d, Eigen::aligned_allocator<Vector6d> > coefficients;

  double startTime;
  double endTime;
  Eigen::VectorXd goalPos;
  Eigen::VectorXd goalVel;
  Eigen::VectorXd goalAcc;
};


} //end namespace
