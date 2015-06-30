#include "CanonicalSystem.h"
#include <cmath>
#include <cassert>
namespace dmp {

using namespace std;

CanonicalSystem::CanonicalSystem(const int numPhases, const double executionTime,
                        const double alpha) :
                        executionTime(executionTime),
                        dt(executionTime/(numPhases - 1)),
                        numPhases(numPhases),
                        alpha(alpha),
                        s0(1.0)
{
  assert(numPhases > 0);
  assert(dt <= executionTime);
}


CanonicalSystem::CanonicalSystem(const double executionTime, const double alpha,
                                 const double dt) :
                        executionTime(executionTime),
                        dt(dt),
                        numPhases((int)(executionTime/dt) + 1),
                        alpha(alpha),
                        s0(1.0)
{
  assert(numPhases > 0);
  assert(dt <= executionTime);

  //assert that the executionTime is approximately divisible by dt
  //if this is not the case the final phase value may not be reached
  assert(abs(((numPhases -1) * dt) - executionTime) < 0.05);
}




double CanonicalSystem::getExecutionTime() const {
  return executionTime;
}
/**
 * Get the phase variable s at a given time t
 */
double CanonicalSystem::getPhase(const double t) const {
  const double b = std::max(1 - alpha * dt/executionTime, 1e-10);
  return pow(b, t / dt) * s0;
}

/**
 * Get the time variable t at the given phase s
 */
double CanonicalSystem::getTime(const double s) const {
  if(s <= 0.0) {
    //the calculation does not work with s == 0 because log(0) is undefined
    return executionTime;
  }
  return dt * (log(s/s0)) / log(1 - alpha * dt / executionTime);
}

double CanonicalSystem::getDt() const {
  return dt;
}

double CanonicalSystem::getAlpha() const {
  return alpha;
}

unsigned CanonicalSystem::getNumberOfPhases() const {
  return numPhases;
}

double CanonicalSystem::calculateAlpha(const double lastPhaseValue,
    const double dt, const double executionTime)
{
  const int numPhases = (int)(executionTime/dt) + 1;
  //assert that the executionTime is approximately divisible by dt
  assert(abs(((numPhases -1) * dt) - executionTime) < 0.05);
  return CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);
}

double CanonicalSystem::calculateAlpha(const double lastPhaseValue,
    const int numPhases) {
  return (1.0 - pow(lastPhaseValue, 1.0 / (numPhases-1))) * (numPhases-1);
}


} //end namespace


