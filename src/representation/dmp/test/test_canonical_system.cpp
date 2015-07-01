#include "catch.hpp"
#include "CanonicalSystem.h"
using namespace dmp;


TEST_CASE("copy ctor ", "[CanonicalSystem]") {
  const double executionTime = 1.2;
  const int numPhases = 14;
  const double lastPhaseValue = 0.01;
  const double dt = executionTime / (numPhases -1);
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, dt, executionTime);
  CanonicalSystem cs1(numPhases, executionTime, alpha);
  CanonicalSystem cs2(cs1);

  REQUIRE(cs1.getDt() == cs2.getDt());
  REQUIRE(cs1.getAlpha() == cs2.getAlpha());
  REQUIRE(cs1.getNumberOfPhases() == cs2.getNumberOfPhases());
  REQUIRE(cs1.getTime(0.5) == cs2.getTime(0.5));
  REQUIRE(cs1.getPhase(0.3) == cs2.getPhase(0.3));

}

TEST_CASE("different init", "[CanonicalSystem]") {
  //regardless of the ctor used the canonical systems should behave in the same way

  const double executionTime = 1.2;
  const int numPhases = 14;
  const double lastPhaseValue = 0.01;
  const double dt = executionTime / (numPhases -1);
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, dt, executionTime);


  CanonicalSystem cs1(numPhases, executionTime, alpha);
  CanonicalSystem cs2(executionTime, alpha, dt);

  for(int i = 0; i < numPhases; ++i)
  {
    const double t = i * dt;
    REQUIRE(cs1.getPhase(t) == Approx(cs2.getPhase(t)));
  }

  for(double s = 1.0; s >= lastPhaseValue; s -= 0.1)
  {
    REQUIRE(cs1.getTime(s) == Approx(cs2.getTime((s))));
  }



}


TEST_CASE("Test Canonical System", "[CanonicalSystem]") {

  const double executionTime = 0.5;
  const int numPhases = 6;
  const double alpha = CanonicalSystem::calculateAlpha(0.01, numPhases);

  CanonicalSystem cs(numPhases, executionTime, alpha);

  REQUIRE(cs.getNumberOfPhases() == numPhases);
  REQUIRE(cs.getPhase(0.0) == 1.0f);
  REQUIRE(cs.getPhase(executionTime) == Approx(0.01));


  SECTION("phase descending") {

    for(unsigned i = 0; i < cs.getNumberOfPhases() - 1; ++i)
    {
      const double t = cs.getDt() * i;
      const double tt = cs.getDt() * (i+1);
      REQUIRE(cs.getPhase(t) > cs.getPhase(tt));
    }
  }


  SECTION("phase time converter") {
    int numSteps = 55;
    const double step = cs.getExecutionTime() / (numSteps - 1);
    for(int i = 0; i < numSteps; ++i)
    {
      double time = i*step;
      double phase = cs.getPhase(time);
      double time2 = cs.getTime(phase);
      REQUIRE( time == Approx(time2));
    }
  }

  SECTION("phase time convert border case") {
    double phase = 0.0;
    double time = cs.getTime(phase);
    REQUIRE(time == Approx(executionTime));
  }

}


