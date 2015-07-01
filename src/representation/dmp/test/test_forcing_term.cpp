/*
 * test_forcing_term.cpp
 *
 *  Created on: Oct 16, 2013
 *      Author: Arne Boeckmann (arne.boeckmann@dfki.de)
 */
#include "catch.hpp"
#include "CanonicalSystem.h"
#include "RbfFunctionApproximator.h"
#include "ForcingTerm.h"
#include <Eigen/Core>
#include <iostream>
#include <limits>
#include <cstdlib>
using namespace dmp;
using namespace Eigen;

TEST_CASE("Test Forcing Term", "[ForcingTerm]") {
  int numPhases = 100;
  double executionTime = 0.8;
  double lastPhaseValue = 0.01;
  double overlap = 0.8;
  int numCenters = 10;
  int taskSpaceDimensions = 2;
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, numPhases);
  CanonicalSystem cs(numPhases, executionTime, alpha);
  RbfFunctionApproximator rbfa(cs, numCenters, overlap);
  ForcingTerm ft(rbfa);

  //create random weights matrix
  //numCenters is equal to number of radial basis functions that is used. We need one weight for each rbf
  MatrixXd weights = MatrixXd::Random(taskSpaceDimensions, numCenters);

  ft.setWeights(weights);

  SECTION("get weights") {
    REQUIRE(weights == ft.getWeights());
  }


  SECTION("output dimension") {
    ArrayXd forces;
    ft.calculateValue(0.3, forces);
    REQUIRE(forces.size() == taskSpaceDimensions);
  }

  SECTION("Equally weighted should sum up to one") {
    double delta = 0.02;
    double weight = rand();
    ft.setWeights(MatrixXd::Constant(taskSpaceDimensions, numCenters, weight));
    for(double t = 0.0; t <= executionTime; t += delta) {
      const double phase = cs.getPhase(t);
      ArrayXd forces;
      ft.calculateValue(phase, forces);
      for(int i = 0; i < taskSpaceDimensions; ++i)
        REQUIRE((forces(i) / weight) == Approx(phase));
    }
  }

  SECTION("Equally weighted should sum up to one (negative weights)") {
    double delta = 0.02;
    double weight =  (-1) * rand();
    ft.setWeights(MatrixXd::Constant(taskSpaceDimensions, numCenters,  weight));
    for(double t = 0.0; t <= executionTime; t += delta) {
      const double phase = cs.getPhase(t);
      ArrayXd forces;
      ft.calculateValue(phase, forces);
      for(int i = 0; i < taskSpaceDimensions; ++i){
        REQUIRE((forces(i) / weight) == Approx(phase));
      }
    }
  }
}


