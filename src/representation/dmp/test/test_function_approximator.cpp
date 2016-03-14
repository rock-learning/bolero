/*
 * test_function_approximator.cpp
 *
 *  Created on: Oct 8, 2013
 *      Author: Arne Boeckmann (arne.boeckmann@dfki.de)
 */
#include <catch/catch.hpp>
#include "CanonicalSystem.h"

#include <vector>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <iostream>
#include <limits>
#define private public
#include "RbfFunctionApproximator.h" //need access to private attributes for testing
#undef private
using namespace dmp;
using namespace std;
using namespace Eigen;


int nearestRbfIndex(unsigned activationIndex, int numCenters, int numPhases) {
  return int(activationIndex*numCenters/float(numPhases));
}

TEST_CASE("copy ctor   ", "[FunctionApproximator]") {
  int numPhases = 10;
  double numCenters = 5;
  double lastPhaseValue = 0.01;
  double executionTime = 0.5;
  double overlap = 0.5;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);
  CanonicalSystem cs(numPhases, executionTime, alpha);
  RbfFunctionApproximator rbfa1(cs, numCenters, overlap);
  RbfFunctionApproximator rbfa2(rbfa1);

  REQUIRE(rbfa1.getCenterCount() == rbfa2.getCenterCount());
  for(int i = 0; i < rbfa1.getCenterCount(); ++i)
  {
    REQUIRE(rbfa1.getCenters()[i] == rbfa2.getCenters()[i]);
  }

  REQUIRE(rbfa1.getWidths().size() == rbfa2.getWidths().size());
}

TEST_CASE("different ctors ", "[FunctionApproximator]") {
  //using different ctors should lead to identical behavior
  int numPhases = 10;
  double numCenters = 5;
  double lastPhaseValue = 0.01;
  double executionTime = 0.5;
  double overlap = 0.5;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);
  CanonicalSystem cs(numPhases, executionTime, alpha);
  RbfFunctionApproximator rbfa1(cs, numCenters, overlap);
  RbfFunctionApproximator rbfa2(rbfa1.centers, rbfa1.widths);

  for(double s = 1.0; s >= lastPhaseValue;  s -= 0.1)
  {
    ArrayXd out1;
    ArrayXd out2;
    rbfa1.getActivations(s, out1);
    rbfa2.getActivations(s, out2);
    REQUIRE(out1.size() == out2.size());
    for(int i = 0; i < out1.size(); ++i)
    {
      REQUIRE(out1[i] == out2[i]);
    }
  }
}


TEST_CASE("activations", "[FunctionApproximator]") {

  int numPhases = 10;
  double numCenters = 5;
  double lastPhaseValue = 0.01;
  double executionTime = 0.5;
  double overlap = 0.2;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  CanonicalSystem cs(numPhases, executionTime, alpha);
  RbfFunctionApproximator rbfa(cs, numCenters, overlap);


  vector<Eigen::ArrayXd, Eigen::aligned_allocator<Eigen::ArrayXd> > activations;
  for(int i = 0; i < numPhases; ++i) {
    activations.push_back(Eigen::ArrayXd());
    const double phase = cs.getPhase(i * cs.getDt());
    rbfa.getActivations(phase, activations.back());
  }

  REQUIRE(activations.size() == numPhases);

  for(unsigned i = 0; i < activations.size(); ++i) {
    REQUIRE(activations[i].size() == numCenters);

    double max = numeric_limits<double>::min();
    for(int j = 0; j < numCenters; ++j) {
      if(activations[i](j) > max) max = activations[i](j);
    }
    //Nearest RBF should have the highest activation.
    REQUIRE(activations[i][nearestRbfIndex(i, numCenters, numPhases)] == Approx(max));

    for(int j = 0; j < numCenters; ++j) {
      //Activations should be within [0.0, 1.0].
      REQUIRE(activations[i][j] >= 0.0);
      REQUIRE(activations[i][j] <= 1.0);
    }
  }
}

TEST_CASE("Overlap", "[FunctionApproximator]") {

  int numPhases = 12;
  int numCenters = 12;
  double lastPhaseValue = 0.01;
  double executionTime = 0.5;
  double overlap = 0.2;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);

  /**
   * This tests checks whether the value of activation (i-1) of phase i is
   * equal to overlap. This is only the case if numCenters is equal to numPhases.
   */
  REQUIRE(numPhases == numCenters);

  CanonicalSystem cs(numPhases, executionTime, alpha);
  RbfFunctionApproximator rbfa(cs, numCenters, overlap);

  vector<Eigen::ArrayXd, Eigen::aligned_allocator<Eigen::ArrayXd> > activations;
  for(int i = 0; i < numPhases; ++i) {
    activations.push_back(Eigen::ArrayXd());
    const double phase = cs.getPhase(i * cs.getDt());
    rbfa.getActivations(phase, activations.back());
  }

  for(unsigned i = 1; i < activations.size(); ++i) {
    REQUIRE(activations[i](i-1) == Approx(overlap));
  }
}

TEST_CASE("Calculate Centers", "[FunctionApproximator]") {

  int numPhases = 12;
  const int numCenters = 14;
  double lastPhaseValue = 0.01;
  double executionTime = 0.5;
  double overlap = 0.2;
  const double dt = executionTime/(numPhases - 1);
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);
  CanonicalSystem cs(numPhases, executionTime, alpha);
  RbfFunctionApproximator rbfa(cs, numCenters, overlap);


  ArrayXd centers;
  ArrayXd widths;
  ArrayXd centers2;
  ArrayXd widths2;
  double centersArr[numCenters];
  double widthsArr[numCenters];
  RbfFunctionApproximator::calculateCenters(cs, numCenters, overlap, centers, widths);
  RbfFunctionApproximator::calculateCenters(lastPhaseValue, executionTime, dt, numCenters,
                                            overlap, centers2, widths2);
  RbfFunctionApproximator::calculateCenters(lastPhaseValue, executionTime, dt, numCenters,
          overlap, centersArr, widthsArr);

  for(int i = 0; i < numCenters; ++i)
  {
    REQUIRE(rbfa.centers(i) == centers(i));
    REQUIRE(rbfa.centers(i) == centers2(i));
    REQUIRE(rbfa.centers(i) == centersArr[i]);
    REQUIRE(rbfa.widths(i) == widths(i));
    REQUIRE(rbfa.widths(i) == widths2(i));
    REQUIRE(rbfa.widths(i) == widthsArr[i]);
  }


}

