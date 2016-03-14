/*
 * test_transformation_system.cpp
 *
 *  Created on: Oct 23, 2013
 *      Author: Arne Boeckmann (arne.boeckmann@dfki.de)
 */
#include <catch/catch.hpp>
#include <iostream>
#include <Eigen/Core>
#include "TransformationSystem.h"
#include "CanonicalSystem.h"
#include "RbfFunctionApproximator.h"
#include "ForcingTerm.h"
#include <memory>
using namespace dmp;
using namespace Eigen;
using namespace std;

typedef RbfFunctionApproximator Rbf;
typedef ForcingTerm Ft;
typedef TransformationSystem Ts;
typedef FifthOrderPolynomial Fop;


/**
 * A very simple test based on hand calculated results
 */
TEST_CASE("Test Transformation System forces", "[TransformationSystem]") {
  const int numPhases = 3;
  const double executionTime = 1.0;
  const double lastPhaseValue = 0.01;
  const double overlap = 0.8;
  const int numCenters = 3;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);
  VectorXd startPos(1); startPos << 0;
  VectorXd startVel(1); startVel << 1;
  VectorXd startAcc(1); startAcc << 0;
  VectorXd endPos(1); endPos << 2;
  VectorXd endVel(1); endVel << 1;
  VectorXd endAcc(1); endAcc << 0;

  ArrayXd fopPos(1);
  ArrayXd fopVel(1);
  ArrayXd fopAcc(1);

  CanonicalSystem cs(numPhases, executionTime, alpha);
  const double dt = cs.getDt();
  Rbf rbfa(cs, numCenters, overlap);
  Ft ft(rbfa);
  Ts ts(ft, executionTime, dt, 25.0, 6.25); //if you change alpha or beta you need to recalculate the asserted values
  Fop fop;
  fop.setConstraints(startPos, startVel, startAcc, endPos, endVel, endAcc, 0, executionTime); //starting phase is always 1.0 => starting time is always 0
  ArrayXXd positions(1, 3);
  positions << 0 , 1 , 2;
  ArrayXXd velocities(1, 3);
  velocities << 1, 1, 1;
  ArrayXXd accelerations(1, 3);
  accelerations << 0, 0, 0;
  ArrayXXd forces(1, 3);
  ts.determineForces(positions, velocities, accelerations, forces, executionTime, dt);

  //following results have been calculated by hand
  REQUIRE(forces(0, 0) == Approx(0));
  REQUIRE(forces(0, 1) == Approx(-46.875)); //THIS WILL FAIL IF alpha is changed
  REQUIRE(forces(0, 2) == Approx(0));

}


TEST_CASE("Gradient calculation", "[TransformationSystem]") {

  int numPhases = 20;
  double executionTime = 0.8;
  double lastPhaseValue = 0.01;
  double overlap = 0.8;
  int numCenters = 10;
  int taskSpaceDimensions = 3;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  CanonicalSystem cs(numPhases, executionTime, alpha);
  const double dt = cs.getDt(); //get dt while we still own the cs
  Rbf rbfa(cs, numCenters, overlap);
  Ft ft(rbfa);
  Ts ts(ft, executionTime, dt);

  ArrayXXd positions(taskSpaceDimensions, numPhases);
  ArrayXXd velocities; //empty by default to trigger the approximation inside determineForces
  ArrayXXd accelerations; //empty by default to trigger the approximation inside determineForces
  ArrayXXd forces;

  //Fill positions with test data
  for(int i = 0; i < taskSpaceDimensions; ++i) {
    for(int j = 0; j < numPhases; ++j) {
      positions(i, j) = i*j;
    }
  }

  ts.determineForces(positions, velocities, accelerations, forces, executionTime, dt);
  for(unsigned i = 0; i < positions.cols()-1; ++i)
  {
    ArrayXd nextPos = positions.col(i) + velocities.col(i) * dt;
    ArrayXd nextVel = velocities.col(i) + accelerations.col(i) * dt;
    for(int j = 0; j < positions.rows(); ++j) {
      //element wise comparision is required because Approx() does not support
      //Eigen types
      REQUIRE(positions.col(i+1)(j) == Approx(nextPos(j)));
      REQUIRE(velocities(j, i) == Approx(nextVel(j)));
    }
  }
}

TEST_CASE("Reaches goal", "[TransformationSystem]") {
  //A simple check to determine that the transformation system always
  //reaches the goal
  const int numPhases = 1000;
  const double executionTime = 10.0;
  const double lastPhaseValue = 0.01;
  const double overlap = 0.8;
  const int numCenters = 50;
  const int numTaskSpaceDimensions = 1;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  ArrayXd startPos(1); startPos << 0;
  ArrayXd startVel(1); startVel << 0;
  ArrayXd startAcc(1); startAcc << 0;
  ArrayXd endPos(1); endPos << 10;
  ArrayXd endVel(1); endVel << 0;
  ArrayXd endAcc(1); endAcc << 0;

  CanonicalSystem cs(numPhases, executionTime, alpha);
  const double dt = cs.getDt();
  Rbf rbfa(cs, numCenters, overlap);
  Ft ft(rbfa);
  Ts ts(ft, executionTime, dt);

  //use random forcing term
  MatrixXd weights = MatrixXd::Random(numTaskSpaceDimensions, numCenters) * 20;
  ft.setWeights(weights);

  ts.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);
  VectorXd times = VectorXd::LinSpaced(numPhases, 0, executionTime);

  for(int i = 0; i < times.size(); ++i)
  {
    const double phase = cs.getPhase(times(i));
    ts.executeStep(phase, times(i), startPos, startVel, startAcc);
  }
  //NOTE: use smaller dt, i.e. larger numberOfPhases to get less integration error in the end
  REQUIRE(startPos(0) == Approx(endPos(0)).epsilon(0.1));
  REQUIRE(startVel(0) == Approx(endVel(0)).epsilon(0.1));
  REQUIRE(startAcc(0) == Approx(endAcc(0)).epsilon(0.1));
}



TEST_CASE("Pulls towards goal despite world influence", "[TransformationSystem") {
  //check if the system always pulls towards the goal when the forcing term is zero

  const int numPhases = 422;
  const double executionTime = 42.20;
  const double lastPhaseValue = 0.0001;
  const double overlap = 0.8;
  const int numCenters = 42;
  const int numTaskSpaceDimensions = 1;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  ArrayXd startPos(1); startPos << 0;
  ArrayXd startVel(1); startVel << 0;
  ArrayXd startAcc(1); startAcc << 0;
  ArrayXd endPos(1); endPos << 42;
  ArrayXd endVel(1); endVel << 0;
  ArrayXd endAcc(1); endAcc << 0;

  CanonicalSystem cs(numPhases, executionTime, alpha);
  const double dt = cs.getDt();
  Rbf rbfa(cs, numCenters, overlap);
  Ft ft(rbfa);
  Ts ts(ft, executionTime, dt);

  //use random forcing term
  MatrixXd weights = MatrixXd::Zero(numTaskSpaceDimensions, numCenters);
  ft.setWeights(weights);

  ts.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);
  VectorXd times = VectorXd::LinSpaced(numPhases, 0, executionTime);

  for(int i = 0; i < times.size(); ++i)
  {
    const double t = times(i);
    double phase = cs.getPhase(t);
    if(i == times.size() - 1) phase = 0.0;
    //simulate environmental distortion that points away from the goal
    startPos -= 1;
    const ArrayXd oldPos = startPos;
    ts.executeStep(phase, t, startPos, startVel, startAcc);
    //assert that the new position is always closer to the goal than the
    //distorted one
    REQUIRE(abs(startPos(0) - endPos(0)) <= abs(oldPos(0) - endPos(0)));
  }
}

TEST_CASE("Reaches goal with end vel/acc != 0", "[TransformationSystem]") {
  const int numPhases = 10000;
  const double executionTime = 2;
  const double lastPhaseValue = 0.01;
  const double overlap = 0.8;
  const int numCenters = 20;
  const int numTaskSpaceDimensions = 1;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);
  ArrayXd startPos(1); startPos << 0;
  ArrayXd startVel(1); startVel << 0;
  ArrayXd startAcc(1); startAcc << 0;
  ArrayXd endPos(1); endPos << 21;
  ArrayXd endVel(1); endVel << 42;
  ArrayXd endAcc(1); endAcc << 0;

  CanonicalSystem cs(numPhases, executionTime, alpha);
  const double dt = cs.getDt();
  Rbf rbfa(cs, numCenters, overlap);
  Ft ft(rbfa);
  Ts ts(ft, executionTime, dt);

  //use random forcing term
  MatrixXd weights = MatrixXd::Zero(numTaskSpaceDimensions, numCenters) * 20;
  ft.setWeights(weights);


  ts.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);
  VectorXd times = VectorXd::LinSpaced(numPhases, 0, executionTime);
  for(int i = 0; i < times.size(); ++i)
  {
    const double t = times(i);
    double phase = cs.getPhase(t);
    if(i == times.size() - 1)
    {//we manually set the last phase to 0
      phase = 0.0;
    }
    ts.executeStep(phase, t, startPos, startVel, startAcc);
    //std::cout << startPos << " " << startVel << " " << startAcc << endl;
  }
  //NOTE use smaller dt, i.e. more phases to get less integration error
  REQUIRE(startPos(0) == Approx(endPos(0)).epsilon(0.1));
  REQUIRE(startVel(0) == Approx(endVel(0)).epsilon(0.1));
  REQUIRE(startAcc(0) == Approx(endAcc(0)).epsilon(0.1));
}




