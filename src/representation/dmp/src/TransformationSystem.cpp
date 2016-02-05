#include "TransformationSystem.h"
#include "EigenHelpers.h"

namespace dmp {
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::VectorXd;

TransformationSystem::TransformationSystem(ForcingTerm& ft, const double executionTime,
                                           const double dt, const double alphaZ, const double betaZ) :
                                           forcingTerm(ft), dt(dt), tau(executionTime),
                                           alphaZ(alphaZ), betaZ(betaZ),
                                           initialized(false)
{}

TransformationSystem::TransformationSystem(ForcingTerm& ft, const TransformationSystem& other) :
    forcingTerm(ft), fifthOrderPoly(other.fifthOrderPoly), dt(other.dt), tau(other.tau),
    alphaZ(other.alphaZ), betaZ(other.betaZ), initialized(other.initialized),
    y0(other.y0), y0d(other.y0d), y0dd(other.y0dd)
{}


void TransformationSystem::initialize(const ArrayXd& startPos, const ArrayXd& startVel,
                                      const ArrayXd& startAcc, const ArrayXd& endPos,
                                      const ArrayXd& endVel, const ArrayXd& endAcc) {

  y0 = startPos;
  y0d = startVel;
  y0dd = startAcc;
  z = startVel * tau;
  initialized = true;
  changeGoal(endPos, endVel, endAcc);
}


void TransformationSystem::changeGoal(const ArrayXd& position, const ArrayXd& velocity,
                                      const ArrayXd& acceleration){
  assert(initialized);
  assert(position.size() == y0.size());
  assert(velocity.size() == y0d.size());
  assert(acceleration.size() == y0dd.size());
  goalPos = position;
  goalVel = velocity;
  goalAcc = acceleration;
  fifthOrderPoly.setConstraints(y0, y0d, y0dd, position, velocity, acceleration,
                                0.0, tau); //starting phase is always 1.0 => starting time is always 0.0
}


void TransformationSystem::changeStart(const ArrayXd& position, const ArrayXd& velocity,
                                       const ArrayXd& acceleration){
  assert(initialized);
  assert(position.size() == y0.size());
  assert(velocity.size() == y0d.size());
  assert(acceleration.size() == y0dd.size());
  y0 = position;
  y0d = velocity;
  y0dd = acceleration;
  fifthOrderPoly.setConstraints(y0, y0d, y0dd, position, velocity, acceleration,
                                0.0, tau); //starting phase is always 1.0 => starting time is always 0.0
}



void TransformationSystem::determineForces(const ArrayXXd& positions, ArrayXXd& velocities,
                                           ArrayXXd& accelerations, ArrayXXd& forces,
                                           const double executionTime, const double dt,
                                           const double alphaZ, const double betaZ)
{
  assert(positions.rows() > 0);
  assert(positions.cols() > 0);

  if(velocities.size() == 0){ //need to approximate velocities
    EigenHelpers::gradient(positions, velocities, dt);
  }

  if(accelerations.size() == 0){ //need to approximate accelerations
    EigenHelpers::gradient(velocities, accelerations, dt);
    accelerations.col(accelerations.cols() - 1).setZero();
  }

  //the final acceleration is needs to be zero for the imitation learning to work
  //properly in Muelling DMPs.

  assert(accelerations.col(accelerations.cols() - 1).isZero(10 * std::numeric_limits<double>::epsilon()));

  assert(positions.rows() == accelerations.rows());
  assert(positions.rows() == velocities.rows());
  assert(positions.cols() == accelerations.cols());
  assert(positions.cols() == velocities.cols());

  //following code is equation (9) from [Muelling2012]
  VectorXd startPos(positions.col(0)); //Starting position of the trajectory
  VectorXd startVel(velocities.col(0)); //Starting velocity of the trajectory
  VectorXd startAcc(accelerations.col(0));//starting acceleration of the trajectory
  VectorXd endPos(positions.col(positions.cols()-1));
  VectorXd endVel(velocities.col(velocities.cols()-1));
  VectorXd endAcc(accelerations.col(accelerations.cols()-1));

  FifthOrderPolynomial fop;
  fop.setConstraints(startPos, startVel, startAcc, endPos, endVel, endAcc,
                     0.0, executionTime); //starting time is always 0.0

  const long taskSpaceDimensions = positions.rows();
  const long numPhases = positions.cols();

  forces.resize(taskSpaceDimensions, numPhases);

  ArrayXd pos(taskSpaceDimensions);
  ArrayXd vel(taskSpaceDimensions);
  ArrayXd acc(taskSpaceDimensions);
  const double t = executionTime;
  const double t2 = t * t;

  for(unsigned i = 0; i < numPhases; ++i) { //iterate over all phases
    //get the pos, vel and acc of the fifth order poly for the current phase
    const double currentT = dt * i;
    fop.getValueAt(currentT, pos, vel, acc);
    assert(pos.array().size() == positions.col(i).size());

    forces.col(i) = t2 * accelerations.col(i) - alphaZ *
                       (betaZ * (pos - positions.col(i)) + vel * t
                        - velocities.col(i) * t) - acc * t2;
  }
}


double TransformationSystem::getDt() const{
    return dt;
}

double TransformationSystem::getTau() const{
    return tau;
}

double TransformationSystem::getAlphaZ() const{
    return alphaZ;
}

double TransformationSystem::getBetaZ() const{
    return betaZ;
}


ArrayXd TransformationSystem::computeZd(const double t, const ArrayXd& y, const ArrayXd& z,
                                         const ArrayXd& f, const double tau) const {
   ArrayXd pos(y.size());
   ArrayXd vel(y.size());
   ArrayXd acc(y.size());
   fifthOrderPoly.getValueAt(t, pos, vel, acc);
   return (alphaZ * (betaZ * (pos - y) + vel * tau - z) +
           acc * tau * tau + f) / tau;
 }


void TransformationSystem::setExecutionTime(const double newTime)
{
  tau = newTime;
  if(initialized)
  {//the fifth order polynom needs to be recalculated if tau is changed.
   //this is done inside changeGoal.
    changeGoal(goalPos, goalVel, goalAcc);
  }
}
}//end namespace
