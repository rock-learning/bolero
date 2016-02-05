/*
 * transformation_system.h
 *
 *  Created on: Oct 22, 2013
 *      Author: Arne Boeckmann (arne.boeckmann@dfki.de)
 */

#pragma once
#include <Eigen/Core>
#include <assert.h>
#include "FifthOrderPolynomial.h"
#include "ForcingTerm.h"

namespace dmp {
/**
 *
 * Transformation system described by [Muelling2012]
 *
 * Generates the goal directed trajectory.
 * The trajectory is usually described by two first order differential
 * equations for the acceleration and the velocity.
 *
 * The parameters \p alphaZ (\f$\alpha_z\f$) and \p betaZ (\f$\beta_z\f$) are used to set the dampening of the
 * transformation system.
 * The default values are set up for critical dampening
 * and are \f$\alpha_z=25.0\f$ and \f$\beta_z=\frac{\alpha_z}{4}\f$.
 * Note that only the relation between \f$\alpha_z\f$ and \f$\beta_z\f$ is
 * relevant for critical dampening. You could just as well use any other value,
 * as long as \f$\beta_z\f$ is one fourth of \f$\alpha_z \f$ the system will be critically
 * damped. However different magnitudes of \f$\alpha_z\f$ may lead to slightly
 * different results. Experiments suggest that it might be possible that
 * certain \f$\alpha_z\f$ values can reduce or increase the integration error.
 * However this needs further investigation.
 *
 * \section Theoretical Justification for beta_z=alpha_z/4
 * The relation between \f$\alpha_z\f$ and \f$\beta_z\f$ that is needed for
 * critical dampening is calculated by rearranging the second order dynamical
 * system
 *  \f[
 *  \tau^2 \ddot{y} = \alpha_z(\beta_z(g-y)+\dot{g}\tau - \tau\dot{y}) + \ddot{g}\tau^2 + nf(z)
 *  \f]
 * to
 * \f[
 * 0 = \tau^2(\ddot{y}-\ddot{g}) + \alpha_z\tau(\dot{y}-\dot{g})+\alpha_z\beta_z(y-g)-nf(z)
 * \f]
 * Now the forcing term \f$nf(z)\f$ can be ignored because it reduces to zero over time
 * \f[
 *  0 = \tau^2(\ddot{y}-\ddot{g}) + \alpha_z\tau(\dot{y}-\dot{g})+\alpha_z\beta_z(y-g)
 * \f]
 * Subsequently the system is reduced to
 * \f[
 * m\ddot{x} + d\dot{x} + kx = 0
 * \f]
 * by substituting \f$m=\tau^2\f$, \f$d=\alpha_z\tau\f$, \f$k=\alpha_z\beta_z\f$
 * and \f$(y-g) = x\f$.
 * This equation describes the motion of a damped oscillating mass.
 * \f$d\f$ is the dampening constant, \f$ k \f$ is the spring constant and \f$ m \f$
 * is the mass.
 *
 *
 * Based on this equation we can calculate the undamped angular frequency of
 * the oscillator:
 * \f[
 * w = \sqrt{k/m}
 * \f]
 * and the damping coefficient:
 * \f[
 * \delta = \frac{d}{2m}
 * \f]
 *
 * According to wikipedia the system is critically damped if
 * \f{eqnarray*}{
 *  w &= \delta \\
 *  \frac{d}{2m} &= \sqrt{k/m} \\
 *  \frac{\alpha_z\tau}{2\tau^2} &= \sqrt{\alpha_z\beta_z/\tau^2} \\
 *  \beta_z &= \frac{\alpha_z}{4}
 * \f}
 */
class TransformationSystem
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TransformationSystem(ForcingTerm& ft, const double executionTime, const double dt,
                       const double alphaZ = 25, const double betaZ = 6.25);

  /**Create a new TransformationSystem with the given forcing term
   * based on the given TransformationSystem. */
  TransformationSystem(ForcingTerm& ft, const TransformationSystem& other);

  /**
   * Sets the initial state of the transformation system.
   * Call this before your first call to executeStep or to
   * reinitialize the transformation system.
   * \param position Initial position of the trajectory
   * \param velocity Initial velocity
   * \param acceleration Initial acceleration
   * \param finalPos final position of the trajectory
   * \param finalVel final velocity of the trajectory
   * \param finalAcc has to be zero for Muelling DMPs
   */
  void initialize(const Eigen::ArrayXd& startPos, const Eigen::ArrayXd& startVel,
                  const Eigen::ArrayXd& startAcc, const Eigen::ArrayXd& endPos,
                  const Eigen::ArrayXd& endVel, const Eigen::ArrayXd& endAcc);

  /**
   * Change goal position during execution.
   * \note the transformation system needs to be initialized before
   *       this method has any effect.
   * \param position the new goal position
   * \param velocity the new goal velocity
   * \param acceleration the new goal acceleration
   *
   **/
  void changeGoal(const Eigen::ArrayXd& position, const Eigen::ArrayXd& velocity,
                  const Eigen::ArrayXd& acceleration);

  /**
   * Change start position during execution.
   * \note the transformation system needs to be initialized before
   *       this method has any effect.
   * \param position the new start position
   * \param velocity the new start velocity
   * \param acceleration the new start acceleration
   *
   **/
  void changeStart(const Eigen::ArrayXd& position, const Eigen::ArrayXd& velocity,
                   const Eigen::ArrayXd& acceleration);

  /**
   * Changes the execution time.
   */
  void setExecutionTime(const double newTime);

  /**Execute a step of the DMP.
   * The desired position, velocity and acceleration will be computed by
   * integration of the first order differential equations, i. e. the
   * following values will be computed in step t:
   *
   *   \f{eqnarray}{
   *   f(s) &=& \textrm{response of forcing\_term}\\
   *   C_t &=& \textrm{coupling term for obstacle avoidance}\\
   *   \eta &=& \textrm{amplitude of the movement}\\
   *   \dot{z}_t &=& \textrm{acceleration based on type of transformation system}\\
   *   \dot{y}_t &=& z_{t-1}/\tau\\
   *   \ddot{y}_t &=& \dot{z}_{t}/\tau\\
   *   z_t &=& z_{t-1} + \dot{z}_t \cdot dt\\
   *   y_t &=& y_{t-1} + \dot{y}_t \cdot dt
   *   \f}
   *
   * \param[in,out] position The current position. Will be overwritten with the new position.
   * \param[out] velocity The new velocity.
   * \param[out] acceleration the new acceleration.
   * \param[in] phase The current phase.
   * \param[in] time The time corresponding to the given phase.
   */
  template <class DerivedA, class DerivedB, class DerivedC>
  void executeStep(const double phase, const double time, Eigen::ArrayBase<DerivedA>& position,
                   Eigen::ArrayBase<DerivedB>& velocity, Eigen::ArrayBase<DerivedC>& acceleration);

  /**Same as above but uses a custom  dt*/
  template <class DerivedA, class DerivedB, class DerivedC>
  void executeStep(const double dt, const double phase, const double time,
                   Eigen::ArrayBase<DerivedA>& position, Eigen::ArrayBase<DerivedB>& velocity,
                   Eigen::ArrayBase<DerivedC>& acceleration);

  /**Determine the required forces to generate a demonstrated trajectory.
   * \param[in] numPhases The number of phases.
   * \param[in] positions Multidimensional points on the trajectory.
   *                      Each column describes a position on the trajectory.
   *                      Each row describes a DOF.
   *                      The number of rows has to be equal to the number of
   *                      task space dimensions in the function approximator.
   * \param[in,out] velocities Derivation of positions.
   *                           Will be approximated using gradient() if empty (size == 0).
   * \param[in,out] accelerations Derivation of velocities.
   *                              Will be approximated using gradient() if empty (size == 0).
   *
   * \param[out] forces An array containing the forces. Each column contains the forces for one phase.
   *                    Each row describes one DOF.
   * \note Start velocity and acceleration will be set to the value of the first entry in velocities/accelerations.
   *       End velocity and acceleration will be set to the value of the last entry in velocities/accelerations.
   *
   * \note Muelling DMPs require that the final acceleration is zero. Therefore
   *       this method enforces that this is the case. Keep this in mind when using
   *       this method for a different kind of dmp.
   */
  static void determineForces(const Eigen::ArrayXXd& positions, Eigen::ArrayXXd& velocities,
                              Eigen::ArrayXXd& accelerations, Eigen::ArrayXXd& forces,
                              const double executionTime, const double dt,
                              const double alphaZ = 25.0, const double betaZ = 6.25);

  virtual double getDt() const;

  virtual double getTau() const;

  virtual double getAlphaZ() const;

  virtual double getBetaZ() const;

private:

  /**
    * Computes zd according to formula (8) from [Muelling2012]
    * \param y the current position
    * \param z velocity * tau from last iteration
    * \param f value of the forcing term for phase s
    * \param ct arbitrary coupling term (set to 0 if you don't want it)
    * \param tau The execution time
    * \param t the time of the current phase
    */
  Eigen::ArrayXd computeZd(const double t, const Eigen::ArrayXd& y, const Eigen::ArrayXd& z,
                     const Eigen::ArrayXd& f, const double tau) const;


  ForcingTerm& forcingTerm;
  FifthOrderPolynomial fifthOrderPoly;
  const double dt;/**<Time between two steps */
  double tau; /**<The execution time */
  /** alphaZ and betaZ control whether the system is critical damped.
   * They should usually not be changed.*/
  const double alphaZ;
  const double betaZ;
  bool initialized;

  //following attributes are used inside executeStep
  Eigen::ArrayXd z;//**< (yd * tau) from last frame */
  Eigen::ArrayXd y0; /**< Position at step 0 */
  Eigen::ArrayXd y0d; /**< Velocity at step 0 */
  Eigen::ArrayXd y0dd; /**< Acceleration at step 0 */

  //the goal is buffered because it is needed to recalculate
  //the fifthOrderPoly if the execution time (tau) is changed
  Eigen::ArrayXd goalPos;
  Eigen::ArrayXd goalVel;
  Eigen::ArrayXd goalAcc;

};

template <class DerivedA, class DerivedB, class DerivedC>
void TransformationSystem::executeStep(const double phase, const double time,
                                       Eigen::ArrayBase<DerivedA>& position,
                                       Eigen::ArrayBase<DerivedB>& velocity,
                                       Eigen::ArrayBase<DerivedC>& acceleration) {
  executeStep(dt, phase, time, position, velocity, acceleration);
}
template <class DerivedA, class DerivedB, class DerivedC>
void TransformationSystem::executeStep(const double dt, const double phase,
                                       const double time, Eigen::ArrayBase<DerivedA>& position,
                                       Eigen::ArrayBase<DerivedB>& velocity,
                                       Eigen::ArrayBase<DerivedC>& acceleration)
{
  assert(initialized);
  Eigen::ArrayXd f;
  forcingTerm.calculateValue(phase, f);
  Eigen::ArrayXd zd = computeZd(time, position, z, f, tau);
  velocity = z / tau;
  acceleration = zd / tau; //follows from: yd = (1/tau)*zd
  z += zd * dt;
  position += velocity * dt;
}

}//end namespace

