#include "catch.hpp"
#include <vector>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <iostream>
#define private public
#include "QuaternionTransformationSystem.h"
#include "RbfFunctionApproximator.h"
#include "ForcingTerm.h"
#include "CanonicalSystem.h"
#include "EigenHelpers.h"

using namespace dmp;
using namespace std;
using namespace Eigen;

typedef RbfFunctionApproximator Rbf;
typedef ForcingTerm Ft;
typedef QuaternionTransformationSystem Qts;
typedef CanonicalSystem Cs;



double randomAngle()
{
  return double(rand()) / RAND_MAX / 10.0 + 0.00001;
}

/**
* generates a random trajectory starting at zero with zero start velocity
*/
void generateTrajectory(Qts::QuaternionVector& pos,
                        ArrayXXd& vel, ArrayXXd& acc, const double dt,
                        const int num)
{
  acc.resize(3, num);
  vel.resize(3, num);

  //generate random accelerations
  for(int i = 0; i < num; ++i)
  {
    acc.col(i) = Vector3d(randomAngle(), randomAngle(), randomAngle());
  }

  vel.col(0) = Vector3d(0, 0, 0);
  //integrate to get velocities
  for(int i = 1; i < num; ++i)
  {
    vel.col(i) = vel.col(i - 1) + acc.col(i - 1) * dt;
  }
  assert(pos.size() == 0);
  Quaternion<double> identity;
  identity.setIdentity();
  pos.push_back(identity);
  //integrate to get positions
  for(int i = 1; i < num; ++i)
  {
    Vector3d oldVel = vel.col(i - 1);
    oldVel *= 0.5 * dt;
    double oldW = 0.0;
    Vector3d oldAx(0.0, 0.0, 0.0);
    Quaternion<double> oldVelAsQuaterion;
    if(oldVel.norm() != 0.0)
    {
      const double len = oldVel.norm();
      oldVel.normalize();
      const double w = cos(len);
      const Vector3d v = sin(len) * oldVel;
      oldVelAsQuaterion = Quaternion<double>(w, v.x(), v.y(), v.z());
    }
    else
    {
      oldVelAsQuaterion = identity;
    }
    pos.push_back(oldVelAsQuaterion * pos[i - 1]);
  }
}

TEST_CASE("gradient", "[QuaternionTransformationSystem]") {
  int numPhases = 20;
  double executionTime = 0.8;
  double lastPhaseValue = 0.01;
  double overlap = 0.8;
  int numCenters = 10;
  int taskSpaceDimensions = 3;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  CanonicalSystem cs(numPhases, executionTime, alpha);
  const double dt = cs.getDt();
  Rbf rbfa(cs, numCenters, overlap);
  Ft ft(rbfa);
  Qts ts(ft, executionTime, dt);


  Qts::QuaternionVector rotations;
  Vector3d axis;
  axis << 1, 0, 0;

  rotations.push_back(Quaternion<double>(AngleAxisd(0.1, axis)));
  rotations.push_back(Quaternion<double>(AngleAxisd(0.2, axis)));

  ArrayXXd results;

  ts.gradient(rotations, results, 1);
  REQUIRE(results.col(0)[0] == Approx(0.1));
  REQUIRE(results.col(0)[1] == Approx(0.0));
  REQUIRE(results.col(0)[2] == Approx(0.0));
  REQUIRE(results.col(1)[0] == Approx(0.1));
  REQUIRE(results.col(1)[1] == Approx(0.0));
  REQUIRE(results.col(1)[2] == Approx(0.0));

}


TEST_CASE("gradient zero", "[QuaternionTransformationSystem]") {
  int numPhases = 20;
  double executionTime = 0.8;
  double lastPhaseValue = 0.01;
  double overlap = 0.8;
  int numCenters = 10;
  int taskSpaceDimensions = 3;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  CanonicalSystem cs(numPhases, executionTime, alpha);
  const double dt = cs.getDt();
  Rbf rbfa(cs, numCenters, overlap);
  Ft ft(rbfa);
  Qts ts(ft, executionTime, dt);


  Qts::QuaternionVector rotations;
  Vector3d axis;
  axis << 1, 0, 0;

  rotations.push_back(Quaternion<double>(AngleAxisd(0.1, axis)));
  rotations.push_back(Quaternion<double>(AngleAxisd(0.1, axis)));

  ArrayXXd results;

  ts.gradient(rotations, results, 1);
  REQUIRE(results.col(0)[0] == Approx(0.0));
  REQUIRE(results.col(0)[1] == Approx(0.0));
  REQUIRE(results.col(0)[2] == Approx(0.0));
  REQUIRE(results.col(1)[0] == Approx(0.0));
  REQUIRE(results.col(1)[1] == Approx(0.0));
  REQUIRE(results.col(1)[2] == Approx(0.0));
  REQUIRE(results.cols() == 2);
}



TEST_CASE(" gradient integrate trajectory", "[QuaternionTransformationSystem]") {
  /* 1) define a small trajectory of rotations around the same axis.
     2) use gradient() to get angular velocities of the trajectory
     3) integrate angular  velocities and check if result is the same
        as the last element of the trajectory  */
  int numPhases = 20;
  double executionTime = 0.8;
  double lastPhaseValue = 0.01;
  double overlap = 0.8;
  int numCenters = 10;
  int taskSpaceDimensions = 3;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  CanonicalSystem cs(numPhases, executionTime, alpha);
  const double dt = cs.getDt();
  Rbf rbfa(cs, numCenters, overlap);
  Ft ft(rbfa);
  Qts ts(ft, executionTime, dt);


  Qts::QuaternionVector rotations;
  Vector3d axis = Vector3d(1, 0, 0);
  AngleAxisd goal(5, axis);
  rotations.push_back(Quaternion<double>(AngleAxisd(1, axis)));
  rotations.push_back(Quaternion<double>(AngleAxisd(2, axis)));
  rotations.push_back(Quaternion<double>(AngleAxisd(3, axis)));
  rotations.push_back(Quaternion<double>(AngleAxisd(4, axis)));
  rotations.push_back(Quaternion<double>(goal));

  ArrayXXd results;

  ts.gradient(rotations, results, 1);
  REQUIRE(results.cols() == rotations.size());
  Quaternion<double> rot(rotations[0]);
  for(int i = 1; i < results.cols(); ++i)
  {
    Vector3d vec = results.col(i);
    const double angle = vec.norm();
    vec.normalize();
    const AngleAxisd a(angle, vec);
    const Quaternion<double> q(a);
    rot = a * rot;
  }

  const AngleAxisd result(rot);
  REQUIRE(result.angle() == Approx(goal.angle()));
  REQUIRE(result.axis()[0] == Approx(goal.axis()[0]));
  REQUIRE(result.axis()[1] == Approx(goal.axis()[1]));
  REQUIRE(result.axis()[2] == Approx(goal.axis()[2]));
}

TEST_CASE("gradient position", "[QuaternionTransformationSystem]") {
  int numPhases = 20;
  double executionTime = 0.8;
  double lastPhaseValue = 0.01;
  double overlap = 0.8;
  int numCenters = 10;
  int taskSpaceDimensions = 3;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  CanonicalSystem cs(numPhases, executionTime, alpha);
  const double dt = cs.getDt();
  Rbf rbfa(cs, numCenters, overlap);
  Ft ft(rbfa);
  Qts ts(ft, executionTime, dt);

  Qts::QuaternionVector rotations;
  ArrayXXd vels;
  ArrayXXd accs;
  generateTrajectory(rotations, vels, accs, 1, 5);

  ArrayXXd approxVels;
  ArrayXXd approxAccs;
  ts.gradient(rotations, approxVels, 1);
  REQUIRE(approxVels.cols() == rotations.size());

  //divide first and last velocity by 2 to compensate for the fact
  //that gradient() does forward/backward quotient for those elements
  approxVels.col(0) /= 2.0;
  approxVels.col(approxVels.cols() - 1) /= 2;

  Quaternion<double> pos(rotations[0]);
  for(int i = 0; i < approxVels.cols(); ++i)
  {
    Vector3d vel = approxVels.col(i);
    const double angle = vel.norm();
    if(angle == 0)
    {
      vel = Vector3d(0, 0, 1);
    }
    else
    {
      vel.normalize();
    }
    const AngleAxisd velAsAA(angle, vel);
    const Quaternion<double> velAsQuaternion(velAsAA);
    pos = velAsQuaternion * pos;
  }
  REQUIRE(pos.w() == Approx(rotations.back().w()));
  REQUIRE(pos.vec().x() == Approx(rotations.back().vec().x()).epsilon(0.01));
  REQUIRE(pos.vec().y() == Approx(rotations.back().vec().y()).epsilon(0.01));
  REQUIRE(pos.vec().z() == Approx(rotations.back().vec().z()).epsilon(0.01));
}

TEST_CASE("gradient angular velocity", "[QuaternionTransformationSystem]") {
  int numPhases = 20;
  double executionTime = 0.8;
  double lastPhaseValue = 0.01;
  double overlap = 0.8;
  int numCenters = 10;
  int taskSpaceDimensions = 3;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  CanonicalSystem cs(numPhases, executionTime, alpha);
  const double dt = cs.getDt();
  Rbf rbfa(cs, numCenters, overlap);
  Ft ft(rbfa);
  Qts ts(ft, executionTime, dt);

  Qts::QuaternionVector rotations;
  ArrayXXd vels;
  ArrayXXd accs;
  generateTrajectory(rotations, vels, accs, 1, 5);

  ArrayXXd approxVels;
  ArrayXXd approxAccs;
  EigenHelpers::gradient(vels, approxAccs, 1);
  REQUIRE(approxAccs.cols() == rotations.size());

  Array3d vel = vels.col(0);
  //divide first and last acceleration by 2 to compensate for the fact
  //that gradient() does forward/backward quotient for those elements
  approxAccs.col(0) /= 2;
  approxAccs.col(approxAccs.cols() - 1) /= 2;
  for(size_t i = 0; i < rotations.size(); ++i)
  {
    vel += approxAccs.col(i);
  }
  REQUIRE(vel[0] == Approx(vels.col(vels.cols() - 1)[0]));
  REQUIRE(vel[1] == Approx(vels.col(vels.cols() - 1)[1]));
  REQUIRE(vel[2] == Approx(vels.col(vels.cols() - 1)[2]));
}

TEST_CASE("qLog", "[QuaternionTransformationSystem]") {
  int numPhases = 20;
  double executionTime = 0.8;
  double lastPhaseValue = 0.01;
  double overlap = 0.8;
  int numCenters = 10;
  int taskSpaceDimensions = 3;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  CanonicalSystem cs(numPhases, executionTime, alpha);
  const double dt = cs.getDt();
  Rbf rbfa(cs, numCenters, overlap);
  Ft ft(rbfa);
  Qts ts(ft, executionTime, dt);


  //quaternion with zero length vector part should result in [0,0,0]
  Array3d zeroResult = ts.qLog(Quaterniond(0.3, 0, 0, 0));
  REQUIRE(zeroResult[0] == 0.0);
  REQUIRE(zeroResult[1] == 0.0);
  REQUIRE(zeroResult[2] == 0.0);


  //quaternion with non zero vector parth should result in
  // arccos(v) * u/|u|
  Vector3d axis(1, 3, 0.37);
  axis.normalize();
  AngleAxisd aa(2, axis);
  Quaterniond q(aa);
  Vector3d vec = q.vec();
  vec.normalize();
  Array3d result = ts.qLog(q);
  Array3d expectedResult = vec * acos(q.w());

  REQUIRE(result[0] == expectedResult[0]);
  REQUIRE(result[1] == expectedResult[1]);
  REQUIRE(result[2] == expectedResult[2]);
}

TEST_CASE("determine forces", "[QuaternionTransformationSystem]") {
  int numPhases = 20;
  double executionTime = 0.8;
  double lastPhaseValue = 0.01;
  double overlap = 0.8;
  int numCenters = 10;
  int taskSpaceDimensions = 3;
  const double alphaZ = 25.0;
  const double betaZ = 6.25;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  CanonicalSystem cs(numPhases, executionTime, alpha);
  const double dt = cs.getDt();

  Vector3d v0(1, 0, 0);
  Vector3d v1(0, 1, 0);
  Vector3d v2(0, 0, 1);

  AngleAxisd aa0(1, v0);
  AngleAxisd aa1(3, v1);
  AngleAxisd aa2(2, v2);

  Quaternion<double> q0(aa0);//w=0.877583, x=0.479426, y=0, z=0
  Quaternion<double> q1(aa1);//w=0.0707372, x=0; y=0.997495; z=0
  Quaternion<double> q2(aa2);//w=0.540302, x=0; y=0; z=0.841471

  Qts::QuaternionVector qs;
  qs.push_back(q0);
  qs.push_back(q1);
  qs.push_back(q2);

  ArrayXXd velocities;
  ArrayXXd accelerations;

  //assuming that the gradient function works, it has been tested above
  Qts::gradient(qs, velocities, dt);
  EigenHelpers::gradient(velocities, accelerations, dt);

  /*
   2 * log(q2 * q0.conjugate()) = [-0.316802, -0.493391, 0.903145]
   2 * log(q2 * q1.conjugate()) = [1.28732, -0.826579, 0.0912902]
   2 * log(q2 * q2.conjugate()) = [0, 0, 0]
   */
  Array3d logQ2Q0(-0.316802, -0.493391, 0.903145);
  Array3d logQ2Q1(1.28732, -0.826579, 0.0912902);
  Array3d logQ2Q2(0, 0, 0);

  ArrayXXd handCalculatedForces(3,3);
  const double executionTime2 = executionTime * executionTime;
  handCalculatedForces.col(0) = executionTime2 * accelerations.col(0) -
                                (alphaZ * (betaZ * 2 * logQ2Q0 -
                                executionTime * velocities.col(0)));

  handCalculatedForces.col(1) = executionTime2 * accelerations.col(1) -
                                (alphaZ * (betaZ * 2 * logQ2Q1 -
                                executionTime * velocities.col(1)));

  handCalculatedForces.col(2) = executionTime2 * accelerations.col(2) -
          (alphaZ * (betaZ * 2 * logQ2Q2 -
                  executionTime * velocities.col(2)));

  ArrayXXd forces;
  Qts::determineForces(qs, velocities, accelerations, forces, dt, executionTime);

  REQUIRE(forces.col(0)[0] == Approx(handCalculatedForces.col(0)[0]));
  REQUIRE(forces.col(0)[1] == Approx(handCalculatedForces.col(0)[1]));
  REQUIRE(forces.col(0)[2] == Approx(handCalculatedForces.col(0)[2]));

  REQUIRE(forces.col(1)[0] == Approx(handCalculatedForces.col(1)[0]));
  REQUIRE(forces.col(1)[1] == Approx(handCalculatedForces.col(1)[1]));
  REQUIRE(forces.col(1)[2] == Approx(handCalculatedForces.col(1)[2]));

  REQUIRE(forces.col(2)[0] == Approx(handCalculatedForces.col(2)[0]));
  REQUIRE(forces.col(2)[1] == Approx(handCalculatedForces.col(2)[1]));
  REQUIRE(forces.col(2)[2] == Approx(handCalculatedForces.col(2)[2]));
}


TEST_CASE("vecExp", "[QuaternionTransformationSystem]")
{
  int numPhases = 20;
  double executionTime = 0.8;
  double lastPhaseValue = 0.01;
  double overlap = 0.8;
  int numCenters = 10;
  int taskSpaceDimensions = 3;
  const double alpha = Cs::calculateAlpha(lastPhaseValue, numPhases);

  Cs cs(numPhases, executionTime, alpha);
  const double dt = cs.getDt();
  Rbf rbfa(cs, numCenters, overlap);
  Ft ft(rbfa);
  Qts ts(ft, executionTime, dt);

  //should be [0,0,0] if input array has length 0
  const Quaterniond id = Quaterniond::Identity();
  const Quaterniond result = ts.vecExp(Array3d(0, 0, 0));
  REQUIRE(result.w() == id.w());
  REQUIRE(result.vec().x() == id.vec().x());
  REQUIRE(result.vec().y() == id.vec().y());
  REQUIRE(result.vec().z() == id.vec().z());


  const Array3d input(1, 2, -3);
  const Quaterniond result2 = ts.vecExp(input);
  const double inputLen = input.matrix().norm();
  const Array3d normalizedInput = input.matrix().normalized();

  REQUIRE(result2.w() == Approx(cos(inputLen)));
  REQUIRE(result2.vec()[0] == Approx(normalizedInput[0] * sin(inputLen)));
  REQUIRE(result2.vec()[1] == Approx(normalizedInput[1] * sin(inputLen)));
  REQUIRE(result2.vec()[2] == Approx(normalizedInput[2] * sin(inputLen)));
}


TEST_CASE("reaches goal", "[QuaternionTransformationSystem]")
{
  const int numPhases = 100;
  const double executionTime = 1.0;
  const double lastPhaseValue = 0.01;
  const double overlap = 0.8;
  const int numCenters = 50;
  const double alpha = Cs::calculateAlpha(lastPhaseValue, numPhases);

  Cs cs(numPhases, executionTime, alpha);
  Rbf rbfa(cs, numCenters, overlap);
  Ft ft(rbfa);
  Qts ts(ft, executionTime, cs.getDt());

  //number of task space dimensions is always 3 for Quaternions.
  MatrixXd weights = MatrixXd::Random(3, numCenters) * 20;
  ft.setWeights(weights);
  const Quaterniond startPos = Quaterniond::Identity();
  const Array3d startVel = Array3d::Zero();
  const AngleAxisd endPosAA(0.42, Vector3d::UnitX());
  const Quaterniond endPos(endPosAA);
  ts.initialize(startPos, startVel, endPos);

  Quaterniond current = startPos;
  VectorXd times = VectorXd::LinSpaced(numPhases, 0, executionTime);
  for(int i = 0; i < times.size(); ++i)
  {
    const double phase = cs.getPhase(times(i));
    ts.executeStep(phase, current);
  }

  REQUIRE(current.w() == Approx(endPos.w()));
  REQUIRE(current.vec()[0] == Approx(endPos.vec()[0]).epsilon(0.1));
  REQUIRE(current.vec()[1] == Approx(endPos.vec()[1]).epsilon(0.1));
  REQUIRE(current.vec()[2] == Approx(endPos.vec()[2]).epsilon(0.1));
}

TEST_CASE("initial velocity" "[QuaternionTransformationSystem]")
{
  /*After three steps a system with initial velocity should have moved further than
  * one without initial velocity*/
  const int numPhases = 100;
  const double executionTime = 1.0;
  const double lastPhaseValue = 0.01;
  const double overlap = 0.8;
  const int numCenters = 50;
  const double alpha = Cs::calculateAlpha(lastPhaseValue, numPhases);

  Cs cs(numPhases, executionTime, alpha);
  Rbf rbfa(cs, numCenters, overlap);
  Ft ft(rbfa);
  Qts ts1(ft, executionTime, cs.getDt());
  Qts ts2(ft, executionTime, cs.getDt());

  MatrixXd weights = MatrixXd::Random(3, numCenters) * 20;
  ft.setWeights(weights);

  const Quaterniond startPos = Quaterniond::Identity();
  const Array3d startVel(100000, 0, 0);
  const AngleAxisd endPosAA(0.42, Vector3d::UnitX());
  const Quaterniond endPos(endPosAA);

  ts1.initialize(startPos, Array3d::Zero(), endPos);
  ts2.initialize(startPos, startVel, endPos);

  Quaterniond out1 = Quaterniond::Identity();
  Quaterniond out2 = Quaterniond::Identity();

  ts1.executeStep(cs.getPhase(0.0), out1);
  ts1.executeStep(cs.getPhase(0.1), out1);
  ts1.executeStep(cs.getPhase(0.2), out1);
  ts2.executeStep(cs.getPhase(0.0), out2);
  ts2.executeStep(cs.getPhase(0.1), out2);
  ts2.executeStep(cs.getPhase(0.2), out2);

  //calculate distance from start to first and start to second
  const Array3d startToFirst = 2 * ts1.qLog(startPos * out1.conjugate());
  const Array3d startToSecond = 2 * ts2.qLog(startPos * out2.conjugate());

  REQUIRE(startToFirst.matrix().norm() < startToSecond.matrix().norm());

}
