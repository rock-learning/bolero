#include "catch.hpp"
#include <memory>
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <limits>
#include "test_helpers.h"
#include "DMPConfig.h"

#define private public
#include "Dmp.h"
#undef private

#include "DMPWrapper.h"



using namespace std;
using namespace Eigen;
using namespace dmp;

TEST_CASE("copy ctor", "[Dmp]")
{
  const int numPhases = 50;
  const double numCenters = 10;
  const double lastPhaseValue = 0.01;
  const double executionTime = 1.0;
  const double overlap = 0.1;
  const double dt = 0.01;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  Dmp dmp(executionTime, alpha, dt, numCenters, overlap);
  Dmp dmp2(dmp);
  REQUIRE(dmp.currentPhase == dmp2.currentPhase);
  REQUIRE(dmp.cs.getDt() == dt);

}


TEST_CASE("raw get activations", "[Dmp")
{
  const int numPhases = 50;
  const int numCenters = 10.0;
  const double lastPhaseValue = 0.01;
  const double executionTime = 1.0;
  const double overlap = 0.1;
  const double dt = executionTime/(numPhases - 1);
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  Dmp dmp(executionTime, alpha, dt, numCenters, overlap);

  ArrayXd activations;
  double activationsArr[10];


  for(double s = 1.0; s >= lastPhaseValue;  s -= 0.1)
  {
    dmp.getActivations(s, true, activations);
    dmp.getActivations(s, true, &activationsArr[0], 10);
    REQUIRE(activations.size() == 10);
    for(int i = 0; i < activations.size(); ++i)
    {
      REQUIRE(activations[i] == activationsArr[i]);
    }

    dmp.getActivations(s, false, activations);
    dmp.getActivations(s, false, &activationsArr[0], 10);
    REQUIRE(activations.size() == 10);
    for(int i = 0; i < activations.size(); ++i)
    {
      REQUIRE(activations[i] == activationsArr[i]);
    }
  }
}

TEST_CASE("multi dimensional raw determine forces", "[Dmp]")
{
  //Test that the raw pointer version of determineForces() method returns the same
  //values as the Eigen version. This test also ensures that the raw pointer version
  //correctly uses the column major storage order
  const int numPhases = 50;
  const double numCenters = 10;
  const double lastPhaseValue = 0.01;
  const double executionTime = 1.0;
  const double overlap = 0.1;
  const double dt = executionTime/(numPhases - 1);
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  Dmp dmp(executionTime, alpha, dt, numCenters, overlap);
  Dmp dmp2(executionTime, alpha, dt, numCenters, overlap);

  ArrayXd startPos(2); startPos << 0, 0;
  ArrayXd startVel(2); startVel << 0, 0;
  ArrayXd startAcc(2); startAcc << 0, 0;
  ArrayXd endPos(2); endPos << 10, 10;
  ArrayXd endVel(2); endVel << 0, 0;
  ArrayXd endAcc(2); endAcc << 0, 0;
  dmp.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);
  dmp2.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);

  ArrayXXd positions = ArrayXXd::Random(2, 50);
  ArrayXXd velocities;
  ArrayXXd accelerations;
  ArrayXXd forces;
  dmp.determineForces(positions, velocities, accelerations, forces, executionTime, dt);

  double positionsArr[50][2];
  for(int i = 0; i < positions.cols(); ++i)
  {
    for(int j = 0; j < positions.rows(); ++j)
    {
      positionsArr[i][j] = positions.col(i)[j];
    }
  }
  double forcesArr[50][2]; //50 = numPhases
  dmp2.determineForces((double*)&positionsArr[0][0], NULL, NULL, 2, 50,
                       (double*)&forcesArr[0][0], 2, 50, executionTime, dt);

  for(int i = 0; i < forces.cols(); ++i)
  {
    for(int j = 0; j < forces.rows(); ++j)
    {
      REQUIRE(forcesArr[i][j] == Approx(forces.col(i)[j]));
    }
  }
}

TEST_CASE("raw determine forces", "[Dmp]")
{//test that raw and eigen determine forces return the same result
 //when vel and acc are approximated
  const int numPhases = 50;
  const double numCenters = 10;
  const double lastPhaseValue = 0.01;
  const double executionTime = 1.0;
  const double overlap = 0.1;
  const double dt = executionTime/(numPhases - 1);
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  Dmp dmp(executionTime, alpha, dt, numCenters, overlap);
  Dmp dmp2(executionTime, alpha, dt, numCenters, overlap);

  ArrayXd startPos(1); startPos << 0;
  ArrayXd startVel(1); startVel << 0;
  ArrayXd startAcc(1); startAcc << 0;
  ArrayXd endPos(1); endPos << 10;
  ArrayXd endVel(1); endVel << 0;
  ArrayXd endAcc(1); endAcc << 0;
  dmp.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);
  dmp2.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);

  ArrayXXd positions = ArrayXXd::Random(1, 50);
  ArrayXXd velocities;
  ArrayXXd accelerations;
  ArrayXXd forces;
  dmp.determineForces(positions, velocities, accelerations, forces, executionTime, dt);

  double positionsArr[1][50];
  for(int i = 0; i < positions.rows(); ++i)
  {
    for(int j = 0; j < positions.cols(); ++j)
    {
      positionsArr[i][j] = positions.row(i)[j];
    }
  }
  double forcesArr[1][50]; //50 = numPhases
  dmp2.determineForces((double*)&positionsArr[0][0], NULL, NULL, 1, 50,
                       (double*)&forcesArr[0][0], 1, 50, executionTime, dt);

  for(int i = 0; i < forces.rows(); ++i)
  {
    for(int j = 0; j < forces.cols(); ++j)
    {
      REQUIRE(forcesArr[i][j] == Approx(forces.row(i)[j]));
    }
  }
}

TEST_CASE("raw determine forces 2", "[Dmp]")
{//test that raw and eigen determine forces return the same result
  //when vel and acc are NOT approximated
  const int numPhases = 50;
  const double numCenters = 10;
  const double lastPhaseValue = 0.01;
  const double executionTime = 1.0;
  const double overlap = 0.1;
  const double dt = executionTime/(numPhases - 1);
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  Dmp dmp(executionTime, alpha, dt, numCenters, overlap);
  Dmp dmp2(executionTime, alpha, dt, numCenters, overlap);

  ArrayXd startPos(2); startPos << 0, 0;
  ArrayXd startVel(2); startVel << 0, 0;
  ArrayXd startAcc(2); startAcc << 0, 0;
  ArrayXd endPos(2); endPos << 10, 10;
  ArrayXd endVel(2); endVel << 0, 0;
  ArrayXd endAcc(2); endAcc << 0, 0;
  dmp.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);
  dmp2.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);

  ArrayXXd positions = ArrayXXd::Random(2, 50);
  ArrayXXd velocities = ArrayXXd::Random(2, 50);
  ArrayXXd accelerations = ArrayXXd::Random(2, 50);
  accelerations.col(49).setZero();
  ArrayXXd forces;
  dmp.determineForces(positions, velocities, accelerations, forces, executionTime, dt);

  double positionsArr[50][2];
  double velocitiesArr[50][2];
  double accelerationsArr[50][2];
  for(int i = 0; i < positions.cols(); ++i)
  {
    for(int j = 0; j < positions.rows(); ++j)
    {
      positionsArr[i][j] = positions.col(i)[j];
      velocitiesArr[i][j] = velocities.col(i)[j];
      accelerationsArr[i][j] = accelerations.col(i)[j];
    }
  }
  double forcesArr[50][2]; //50 = numPhases
  dmp2.determineForces((double*)&positionsArr[0][0], (double*)&velocitiesArr[0][0],
                       (double*)&accelerationsArr[0][0], 2, 50, (double*)&forcesArr[0][0], 2, 50,
                        executionTime, dt);

  for(int i = 0; i < forces.cols(); ++i)
  {
    for(int j = 0; j < forces.rows(); ++j)
    {
      REQUIRE(forcesArr[i][j] == Approx(forces.col(i)[j]));
    }
  }
}



TEST_CASE("raw executeStep", "[Dmp]")
{
  const int numPhases = 50;
  const double numCenters = 10;
  const double lastPhaseValue = 0.01;
  const double executionTime = 1.0;
  const double overlap = 0.1;
  const double dt = executionTime/(numPhases - 1);
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  Dmp dmp(executionTime, alpha, dt, numCenters, overlap);
  Dmp dmp2(executionTime, alpha, dt, numCenters, overlap);

  ArrayXd startPos(1); startPos << 0;
  ArrayXd startVel(1); startVel << 0;
  ArrayXd startAcc(1); startAcc << 0;
  ArrayXd endPos(1); endPos << 10;
  ArrayXd endVel(1); endVel << 0;
  ArrayXd endAcc(1); endAcc << 0;
  dmp.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);
  dmp2.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);

  const ArrayXXd weights = ArrayXXd::Random(1, numCenters) * 20;
  dmp.setWeights(weights);
  dmp2.setWeights(weights);


  ArrayXd pos(1); startPos << 0;
  ArrayXd vel(1); startVel << 0;
  ArrayXd acc(1); startAcc << 0;

  double posArr[1] = {0};
  double velArr[1] = {0};
  double accArr[1] = {0};

  //executing both dmps should return the same results
  for(int i = 0; i < numPhases; ++i)
  {
    dmp.executeStep(pos, vel, acc);
    dmp2.executeStep(&posArr[0], &velArr[0], &accArr[0], 1);
    REQUIRE(pos[0] == Approx(posArr[0]));
    REQUIRE(vel[0] == Approx(velArr[0]));
    REQUIRE(acc[0] == Approx(accArr[0]));
  }
}

TEST_CASE("raw initialize", "[Dmp]")
{

  const int numPhases = 50;
  const double numCenters = 10;
  const double lastPhaseValue = 0.01;
  const double executionTime = 1.0;
  const double overlap = 0.1;
  const double dt = executionTime/(numPhases - 1);
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  Dmp dmp(executionTime, alpha, dt, numCenters, overlap);
  Dmp dmp2(executionTime, alpha, dt, numCenters, overlap);

  ArrayXd startPos(1); startPos << 0;
  ArrayXd startVel(1); startVel << 0;
  ArrayXd startAcc(1); startAcc << 0;
  ArrayXd endPos(1); endPos << 10;
  ArrayXd endVel(1); endVel << 0;
  ArrayXd endAcc(1); endAcc << 0;
  dmp.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);

  double startPosArr[1] = {0};
  double startVelArr[1] = {0};
  double startAccArr[1] = {0};
  double endPosArr[1] = {10};
  double endVelArr[1] = {0};
  double endAccArr[1] = {0};

  dmp2.initialize(&startPosArr[0], &startVelArr[0], &startAccArr[0],
                  &endPosArr[0], &endVelArr[0], &endAccArr[0], 1);


  const ArrayXXd weights = ArrayXXd::Random(1, numCenters) * 20;
  dmp.setWeights(weights);
  dmp2.setWeights(weights);


  ArrayXd pos(1); startPos << 0;
  ArrayXd vel(1); startVel << 0;
  ArrayXd acc(1); startAcc << 0;

  ArrayXd pos2(1); startPos << 0;
  ArrayXd vel2(1); startVel << 0;
  ArrayXd acc2(1); startAcc << 0;

  //executing both dmps should return the same results
  for(int i = 0; i < numPhases; ++i)
  {
    dmp.executeStep(pos, vel, acc);
    dmp2.executeStep(pos2, vel2, acc2);
    REQUIRE(pos[0] == Approx(pos2[0]));
    REQUIRE(vel[0] == Approx(vel2[0]));
    REQUIRE(acc[0] == Approx(acc2[0]));
  }

}

TEST_CASE("raw changeGoal", "[Dmp]")
{
  const int numPhases = 50;
  const double numCenters = 10;
  const double lastPhaseValue = 0.01;
  const double executionTime = 1.0;
  const double overlap = 0.1;
  const double dt = executionTime/(numPhases - 1);
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  Dmp dmp(executionTime, alpha, dt, numCenters, overlap);

  double pos[1];
  double vel[1];
  double acc[1];
  pos[0] = 41.0;
  vel[0] = 42.0;
  acc[0] = 43.0;

  ArrayXd startPos(1); startPos << 0;
  ArrayXd startVel(1); startVel << 0;
  ArrayXd startAcc(1); startAcc << 0;
  ArrayXd endPos(1); endPos << 10;
  ArrayXd endVel(1); endVel << 0;
  ArrayXd endAcc(1); endAcc << 0;
  dmp.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);

  dmp.changeGoal(&pos[0], &vel[0], &acc[0], 1);
  REQUIRE(dmp.endPos[0] == Approx(pos[0]));
  REQUIRE(dmp.endVel[0] == Approx(vel[0]));
  REQUIRE(dmp.endAcc[0] == Approx(acc[0]));
}


TEST_CASE("raw getPhases", "[Dmp]")
{
  const int numPhases = 50;
  const double numCenters = 10;
  const double lastPhaseValue = 0.01;
  const double executionTime = 1.0;
  const double overlap = 0.1;
  const double dt = executionTime/(numPhases - 1);
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  Dmp dmp(executionTime, alpha, dt, numCenters, overlap);

  ArrayXd phases(numPhases);
  dmp.getPhases(phases);

  double rawPhases[numPhases];
  for(int i = 0; i < numPhases; ++i)
  {
    rawPhases[i] = 42.0;
  }

  dmp.getPhases(&rawPhases[0], numPhases);

  for(int i = 0; i < numPhases; ++i)
  {
    REQUIRE(rawPhases[i] == Approx(phases[i]));
  }

}

TEST_CASE("raw setWeights", "[Dmp]")
{

  const int numPhases = 50;
  const double numCenters = 10;
  const double lastPhaseValue = 0.01;
  const double executionTime = 1.0;
  const double overlap = 0.1;
  const double dt = executionTime/(numPhases - 1);
  const int taskSpaceDimensions = 3;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  Dmp dmp(executionTime, alpha, dt, numCenters, overlap);

  const int rows = taskSpaceDimensions;
  const int cols = numCenters;
  double weights[cols][rows];

  for(int i = 0; i < cols; ++i)
  {
    for(int j = 0; j < rows; ++j)
    {
      weights[i][j] = 42;
    }
  }

  dmp.setWeights(const_cast<const double*>(&weights[0][0]), rows, cols);

  for(int i = 0; i < cols; ++i)
  {
    for(int j = 0; j < rows; ++j)
    {
      REQUIRE(dmp.ft.getWeights().col(i)[j] == weights[i][j]);
    }
  }

}

TEST_CASE("executeStep return", "[Dmp]")
{
  const int numPhases = 50;
  const double numCenters = 10;
  const double lastPhaseValue = 0.01;
  const double executionTime = 1.0;
  const double overlap = 0.1;
  const double dt = executionTime/(numPhases - 1);
  const int taskSpaceDimensions = 1;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  Dmp dmp(executionTime, alpha, dt, numCenters, overlap);
  REQUIRE(dmp.cs.getNumberOfPhases() == numPhases);


  ArrayXd startPos(1); startPos << 0;
  ArrayXd startVel(1); startVel << 0;
  ArrayXd startAcc(1); startAcc << 0;
  ArrayXd endPos(1); endPos << 10;
  ArrayXd endVel(1); endVel << 0;
  ArrayXd endAcc(1); endAcc << 0;
  dmp.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);

  const ArrayXXd weights = ArrayXXd::Random(taskSpaceDimensions, numCenters) * 20;
  dmp.setWeights(weights);

  unsigned i = 0;
  do
  {
    ++i;
  }while(dmp.executeStep(startPos, startVel, startAcc));

  REQUIRE(i == numPhases -1);


  startPos << 0;
  startVel << 0;
  startAcc << 0;
  dmp_cpp::DMPWrapper wrapper;
  wrapper.init_from_dmp(dmp);
  wrapper.dmp().initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);
  i = 0;
  do
  {
    ++i;
  }while(wrapper.dmp().executeStep(startPos, startVel, startAcc));
  REQUIRE(i == numPhases -1);
}

TEST_CASE("getPhases","[Dmp]")
{
  const int numPhases = 50;
  const double numCenters = 10;
  const double lastPhaseValue = 0.01;
  const double executionTime = 1.0;
  const double overlap = 0.1;
  const double dt = executionTime/(numPhases - 1);
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  Dmp dmp(executionTime, alpha, dt, numCenters, overlap);

  ArrayXd phases(numPhases);
  dmp.getPhases(phases);

  REQUIRE(phases[0] == Approx(1.0));
  REQUIRE(phases[phases.size()-1] == Approx(lastPhaseValue));
}


TEST_CASE("changeTime", "[Dmp]")
{
  const int numPhases = 50;
  const double numCenters = 10;
  const double lastPhaseValue = 0.01;
  double executionTime = 1.0;
  const double overlap = 0.1;
  const double dt = executionTime/(numPhases - 1);
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  Dmp dmp(executionTime, alpha, dt, numCenters, overlap);
  dmp.changeTime(2.0);
  //if we double the execution time without changing dt the
  //number of phases should double
  REQUIRE(dmp.cs.getNumberOfPhases() == 2 * numPhases - 1);

}

TEST_CASE("integration granularity", "[Dmp]")
{
  const int numPhases = 10;
  const double numCenters = 35;
  const double lastPhaseValue = 0.01;
  double executionTime = 1.5;
  const double overlap = 0.1;
  const double dt = executionTime/(numPhases - 1);
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);
  const double alphaZ = 25.0;
  const double betaZ = 6.25;

  const ArrayXXd weights = ArrayXXd::Random(1, numCenters) * 20;
  ArrayXd startPos(1);
  ArrayXd startVel(1);
  ArrayXd startAcc(1);
  ArrayXd endPos(1); endPos << 42;
  ArrayXd endVel(1); endVel << 1.0;
  ArrayXd endAcc(1); endAcc << 0;

  //create dmps with different integration granularity.
  //each dmp should have less error than the one before
  double oldError[3];
  oldError[0] = std::numeric_limits<double>::max();
  oldError[1] = std::numeric_limits<double>::max();
  oldError[2] = std::numeric_limits<double>::max();

  for(int i = 1; i < 100; ++i)
  {
    Dmp dmp(executionTime, alpha, dt, numCenters, overlap, alphaZ, betaZ, i);
    startPos << -5;
    startVel << 0;
    startAcc << 0;
    dmp.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);
    dmp.setWeights(weights);

    while(dmp.executeStep(startPos, startVel, startAcc))
    {}

    double error[3];
    error[0] = abs(endPos[0] - startPos[0]);
    error[1] = abs(endVel[0] - startVel[0]);
    error[2] = abs(endAcc[0] - startAcc[0]);
    REQUIRE(oldError[0] >= error[0]);
    REQUIRE(oldError[1] >= error[1]);
    REQUIRE(oldError[2] >= error[2]);
    for(int i = 0; i < 3; ++i) oldError[i] = error[i];

  }

}


TEST_CASE("initialize from config", "[Dmp]")
{
  //a dmp initialized from a cfg should have the same behavior
  //as a dmp initialized manually
  using namespace dmp_cpp;

  const int numPhases = 10;
  const double numCenters = 10;
  const double lastPhaseValue = 0.01;
  double executionTime = 1.5;
  const double overlap = 0.1;
  const double dt = executionTime/(numPhases - 1);
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  Dmp dmp(executionTime, alpha, dt, numCenters, overlap);
  Dmp dmp2(executionTime, alpha, dt, numCenters, overlap);


  ArrayXd startPos(1); startPos << 0;
  ArrayXd startVel(1); startVel << 0;
  ArrayXd startAcc(1); startAcc << 0;
  ArrayXd endPos(1); endPos << 10;
  ArrayXd endVel(1); endVel << 0;
  ArrayXd endAcc(1); endAcc << 0;
  dmp.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);



  DMPConfig cfg;
  cfg.config_name = "a name";
  cfg.dmp_execution_time = executionTime;
  cfg.dmp_startPosition.resize(1);
  cfg.dmp_startPosition[0] = 0;
  cfg.dmp_startVelocity.resize(1);
  cfg.dmp_startVelocity[0] = 0;
  cfg.dmp_startAcceleration.resize(1);
  cfg.dmp_startAcceleration[0] = 0;
  cfg.dmp_endPosition.resize(1);
  cfg.dmp_endPosition[0] = 10;
  cfg.dmp_endVelocity.resize(1);
  cfg.dmp_endVelocity[0] = 0;
  cfg.dmp_endAcceleration.resize(1);
  cfg.dmp_endAcceleration[0] = 0;
  cfg.fully_initialized = true;

  dmp2.initialize(cfg);

  const ArrayXXd weights = ArrayXXd::Random(1, numCenters) * 20;
  dmp.setWeights(weights);
  dmp2.setWeights(weights);

  ArrayXd pos(1); pos << 0;
  ArrayXd vel(1); vel << 0;
  ArrayXd acc(1); acc << 0;
  ArrayXd pos2(1); pos2 << 0;
  ArrayXd vel2(1); vel2 << 0;
  ArrayXd acc2(1); acc2 << 0;

  REQUIRE(dmp.cs.getNumberOfPhases() == dmp2.cs.getNumberOfPhases());
  REQUIRE(dmp.cs.getDt() == dmp2.cs.getDt());

  for(unsigned i = 0; i < dmp2.cs.getNumberOfPhases(); ++i)
  {
    dmp.executeStep(pos, vel, acc);
    dmp2.executeStep(pos2, vel2, acc2);

    REQUIRE(pos[0] == Approx(pos2[0]));
    REQUIRE(vel[0] == Approx(vel2[0]));
    REQUIRE(acc[0] == Approx(acc2[0]));
  }

}


TEST_CASE("get tast dim", "[Dmp]")
{
  const int numPhases = 50;
  const double numCenters = 10;
  const double lastPhaseValue = 0.01;
  double executionTime = 1.0;
  const double overlap = 0.1;
  const double dt = executionTime/(numPhases - 1);
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  Dmp dmp(executionTime, alpha, dt, numCenters, overlap);

  ArrayXd startPos(1); startPos << 0;
  ArrayXd startVel(1); startVel << 0;
  ArrayXd startAcc(1); startAcc << 0;
  ArrayXd endPos(1); endPos << 10;
  ArrayXd endVel(1); endVel << 0;
  ArrayXd endAcc(1); endAcc << 0;
  dmp.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);

  REQUIRE(dmp.getTaskDimensions() == 1);

}



TEST_CASE("get current Phase", "[Dmp]")
{
  const int numPhases = 50;
  const double numCenters = 10;
  const double lastPhaseValue = 0.01;
  double executionTime = 1.0;
  const double overlap = 0.1;
  const double dt = executionTime/(numPhases - 1);
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  Dmp dmp(executionTime, alpha, dt, numCenters, overlap);

  ArrayXd startPos(1); startPos << 0;
  ArrayXd startVel(1); startVel << 0;
  ArrayXd startAcc(1); startAcc << 0;
  ArrayXd endPos(1); endPos << 10;
  ArrayXd endVel(1); endVel << 0;
  ArrayXd endAcc(1); endAcc << 0;
  dmp.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);

  const ArrayXXd weights = ArrayXXd::Random(1, numCenters) * 20;
  dmp.setWeights(weights);

  ArrayXd phases;
  dmp.getPhases(phases);

  for(unsigned i = 0; i < dmp.cs.getNumberOfPhases(); ++i)
  {
    const double s = dmp.getCurrentPhase();
    REQUIRE(s == Approx(phases[i]));
    dmp.executeStep(startPos, startVel, startAcc);

  }
}

TEST_CASE("change goal", "[Dmp]")
{
  const int numPhases = 50;
  const double numCenters = 10;
  const double lastPhaseValue = 0.01;
  double executionTime = 1.0;
  const double overlap = 0.1;
  const double dt = executionTime/(numPhases - 1);
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  Dmp dmp(executionTime, alpha, dt, numCenters, overlap);

  ArrayXd startPos(1); startPos << 0;
  ArrayXd startVel(1); startVel << 0;
  ArrayXd startAcc(1); startAcc << 0;
  ArrayXd endPos(1); endPos << 10;
  ArrayXd endVel(1); endVel << 0;
  ArrayXd endAcc(1); endAcc << 0;
  dmp.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);

  const ArrayXXd weights = ArrayXXd::Random(1, numCenters) * 20;
  dmp.setWeights(weights);

  ArrayXd phases;
  dmp.getPhases(phases);
  //change goal after half the phases
  //assert that the new goal is still reached
  for(unsigned i = 0; i < dmp.cs.getNumberOfPhases() / 2; ++i)
  {
    dmp.executeStep(startPos, startVel, startAcc);
  }
  endPos << 5;
  dmp.changeGoal(endPos, endVel, endAcc);

  while(dmp.executeStep(startPos, startVel, startAcc))
  {}
  REQUIRE(startPos[0] == Approx(endPos[0]).epsilon(0.1));
}


TEST_CASE("generate Config", "[Dmp]")
{
  const int numPhases = 50;
  const double numCenters = 10;
  const double lastPhaseValue = 0.01;
  double executionTime = 1.0;
  const double overlap = 0.1;
  const double dt = executionTime/(numPhases - 1);
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  Dmp dmp(executionTime, alpha, dt, numCenters, overlap);

  ArrayXd startPos(1); startPos << 0;
  ArrayXd startVel(1); startVel << 1;
  ArrayXd startAcc(1); startAcc << 2;
  ArrayXd endPos(1); endPos << 3;
  ArrayXd endVel(1); endVel << 4;
  ArrayXd endAcc(1); endAcc << 5;
  dmp.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);

  dmp_cpp::DMPConfig cfg = dmp.generateConfig();

  REQUIRE(cfg.dmp_execution_time == executionTime);
  REQUIRE(cfg.dmp_endAcceleration[0] == endAcc(0));
  REQUIRE(cfg.dmp_startAcceleration[0] == startAcc(0));
  REQUIRE(cfg.dmp_endPosition[0] == endPos(0));
  REQUIRE(cfg.dmp_endVelocity[0] == endVel(0));
  REQUIRE(cfg.dmp_startPosition[0] == startPos(0));
  REQUIRE(cfg.dmp_startVelocity[0] == startVel(0));


}



