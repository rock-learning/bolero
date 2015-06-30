/**
 * @file canonical_system.h
 *
 *  @author Arne Boeckmann (arne.boeckmann@dfki.de)
 */

#pragma once
#include <Eigen/Core>
namespace dmp {


/**
 * Implements the canonical system suggested by [Pastor2009].
 *
 * The DMPs replace real time t by the "phase variable" s. s is determined by the
 * following first-order linear dynamics:
 *
 *  \f[
 *  \tau\dot{s} = -\alpha s
 *  \f]
 *
 *  where \f$\alpha\f$ is defined implicitly by defining the value of s at the time of convergence.
 *  s starts from 1 and converges monotonically to 0.
 *  Avoiding the explicit time dependency has the advantage
 *  that we have obtained an autonomous dynamical system, which can be modified
 *  online with additional coupling terms.
 *
 *  As long as the execution time does not change, the dynamical system can be
 *  solved directly using
 *  \f[
 *    s(t) = exp(\frac{-\alpha}{\tau t}).
 *  \f]
 *  Solving the equation directly provides results with higher precision. Therefore
 *  this implementation uses the direct solution. However changing the execution
 *  time mid-run is not possible with this solution, thus a new instance has
 *  to be created to change the execution time.
 *
 *  @note Be careful when changing the execution time mid-run as changes in the
 *        execution time lead to jumps in the phase value which in turn lead jumps
 *        in the dmp position and extremly high accelerations. If you really need
 *        to change the execution time mid-run you need to interpolate!!
 *
 */
class CanonicalSystem {
public:
  /**
   * \param numPhases Number of steps during the execution of the DMP.
   * \param executionTime Time constant to scale speed.
   *                      Roughly this is movement time until convergence to the goal.
   * \param alpha The decay factor. Ultimately this factor defines what
   *              the value at the last phase will be.
   *              Use CanonicalSystem::calculateAlpha to convert from
   *              last phase value to alpha
   */
  CanonicalSystem(const int numPhases, const double executionTime,
                  const double alpha);

  /**
  * \param executionTime Time constant to scale speed.
  *                      Roughly this is movement time until convergence to the goal.
  * \param alpha The decay factor. Ultimately this factor defines what
  *              the value at the last phase will be.
  *              Use CanonicalSystem::calculateAlpha to convert from
  *              last phase value to alpha
  */
  explicit CanonicalSystem(const double executionTime, const double alpha, const double dt);

  virtual double getExecutionTime() const;

  /** Get the phase variable s at a given time t */
  virtual double getPhase(const double t) const;

  /** Get the time variable t at the given phase s  */
  virtual  double getTime(const double s) const;

  virtual double getDt() const;

  virtual double getAlpha() const;

  virtual unsigned getNumberOfPhases() const;

  /**
   * Generates a list of all phase values.
   * \param[out] the phases
   *
   * \note The phases will be generated on the fly, therefore calling this method
   *       is computationally expensive.
   */
  template <class Derived>
  void getPhases(Eigen::ArrayBase<Derived>& phases) const;

  /**Determines the decay factor alpha.
   * \param lastPhaseValue the value the canonical system should reach at executionTime */
  static double calculateAlpha(const double lastPhaseValue, const double dt,
                               const double executionTime);
  static double calculateAlpha(const double lastPhaseValue, const int numPhases);


private:

  double executionTime; /**<total runtime till convergence*/
  double dt;/**<difference between two steps in seconds */
  int numPhases;
  double alpha; /**<Constant that us used to decide how fast the system descents towards zero */
  double s0;/**<start value of s */
};

template <class Derived>
void CanonicalSystem::getPhases(Eigen::ArrayBase<Derived>& phases) const
{
  //resizing does only work for some children of ArrayBase
  //If it does not work, eigen will die with an assert at this point
  phases.resize(numPhases);
  for(int i = 0; i < numPhases; ++i)
  {
    const double t = getDt() * i;
    phases[i] = getPhase(t);
  }
}

}

