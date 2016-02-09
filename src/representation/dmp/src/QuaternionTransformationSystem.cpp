#include <Eigen/Geometry>
#include "QuaternionTransformationSystem.h"
#include "ForcingTerm.h"
#include "EigenHelpers.h"

namespace dmp
{
using Eigen::Quaterniond;
using Eigen::Array3d;
using Eigen::ArrayXXd;
using Eigen::ArrayXd;
using Eigen::AngleAxisd;
using Eigen::Vector3d;

QuaternionTransformationSystem::QuaternionTransformationSystem(ForcingTerm &ft,
        const double executionTime, const double dt, double const alphaZ,
        double const betaZ) :
        executionTime(executionTime), dt(dt), alphaZ(alphaZ), betaZ(betaZ),
        forcingTerm(ft), initialized(false)

{
}

void QuaternionTransformationSystem::initialize(const Quaterniond& startPos,
                                                const Array3d& startVel,
                                                const Quaterniond& endPos)
{
  this->startPos = startPos;
  this->endPos = endPos;
  eta = executionTime * startVel; //see paragraph below (21) in [Ude2014]
  initialized = true;
}
void QuaternionTransformationSystem::gradient(const QuaternionVector& rotations,
                                              ArrayXXd& velocities,
                                              const double dt, bool allowFinalVelocity)
{
  assert(velocities.rows() == 0);
  assert(velocities.cols() == 0);
  assert(rotations.size() > 1);
  velocities.resize(3, rotations.size());
  const double dt2 = dt * 2;

  //forward difference quotient (special case for first element)
  velocities.col(0) = 2 * qLog(rotations[1] * rotations[0].conjugate()) / dt;
  //central difference quotient
  for(size_t i = 1; i < rotations.size() - 1; ++i)
  {
    const Quaterniond& q0 = rotations[i - 1];
    const Quaterniond& q1 = rotations[i + 1];
    velocities.col(i) = 2 * qLog(q1 * q0.conjugate()) / dt2;
  }

  //backward difference quotient (special case for last element)
  const int last = rotations.size() - 1;
  if(allowFinalVelocity)
    velocities.col(last) = 2 * qLog(rotations[last] * rotations[last - 1].conjugate()) / dt;
  else
    velocities.col(last).setZero();
}

Array3d QuaternionTransformationSystem::qLog(const Quaterniond& q)
{
  const double len = q.vec().norm();
  if(len == 0.0)
  {
    return Array3d::Zero();
  }
  return q.vec().array() / len * acos(q.w());
}

Quaterniond QuaternionTransformationSystem::vecExp(const Vector3d& input) const
{
  const double len = input.norm();
  if(len != 0)
  {
    const Array3d vec = sin(len) * input / len;
    return Quaterniond(cos(len), vec.x(), vec.y(), vec.z());
  }
  else
  {
    return Quaterniond::Identity();
  }
}


void QuaternionTransformationSystem::determineForces(const QuaternionVector &rotations,
                                                     ArrayXXd &velocities,
                                                     ArrayXXd &accelerations,
                                                     ArrayXXd& forces,
                                                     const double dt, const double executionTime,
                                                     const double alphaZ, const double betaZ,
                                                     bool allowFinalVelocity)
{
  assert(rotations.size() > 0);

  if(velocities.size() == 0)
  {
    gradient(rotations, velocities, dt, allowFinalVelocity);
  }
  if(accelerations.size() == 0)
  {
    EigenHelpers::gradient(velocities, accelerations, dt, false);
  }

  assert(rotations.size() == unsigned(velocities.cols()));
  assert(rotations.size() == unsigned(accelerations.cols()));
  assert(accelerations.rows() == 3);
  assert(velocities.rows() == 3);

  //following code is equation (16) from [Ude2014] rearranged to $f_0$
  forces.resize(3, rotations.size());
  for(size_t i = 0; i < rotations.size(); ++i)
  {
    forces.col(i) = executionTime * executionTime * accelerations.col(i) - (alphaZ *
                    (betaZ * 2 * qLog(rotations.back() * rotations[i].conjugate()) -
                    executionTime * velocities.col(i)));
  }
}


void QuaternionTransformationSystem::executeStep(const double phase, Quaterniond &position)
{
  assert(initialized);
  ArrayXd f;
  forcingTerm.calculateValue(phase, f);
  //for Quaternions the forcing term needs to have 3 entries
  assert(f.size() == 3);

  const Array3d log = qLog(endPos * position.conjugate());
  const Array3d etaD = (alphaZ * (betaZ * 2 * log - eta) + f) / executionTime;
  position = vecExp(dt / 2.0 * eta / executionTime) * position;
  eta += etaD * dt;
}
} //end namespace
