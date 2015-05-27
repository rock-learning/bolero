#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include <LoadableBehavior.h>
#include <BLLoader.h>

using namespace bolero;
using namespace std;

TEST_CASE( "init test", "[PyLoadableBehavior]" ) {
  bl_loader::BLLoader loader;
  LoadableBehavior* behav = loader.acquireBehavior("TestBehavior");
  REQUIRE_NOTHROW(behav->initialize("init.yaml"));
  REQUIRE(behav->getNumInputs() == 3);
  REQUIRE(behav->getNumOutputs() == 3);
}

TEST_CASE( "configure and stepping", "[PyLoadableBehavior]" ) {

  bl_loader::BLLoader loader;
  LoadableBehavior* behav = loader.acquireBehavior("TestBehavior");
  REQUIRE_NOTHROW(behav->initialize("init.yaml"));
  REQUIRE_NOTHROW(behav->configure("config.yaml"));

  for(int i = 0; i < 5; ++i)
  {
    REQUIRE(behav->canStep());
    double data[3] = {1, 2, 3}; //the test_behavior expects these inputs
    behav->setInputs(&data[0], 3);
    behav->step();
    behav->getOutputs(&data[0], 3);

    //step multiplies the data with the multiplier specified in the config file, in this case -1
    REQUIRE(data[0] == -1);
    REQUIRE(data[1] == -2);
    REQUIRE(data[2] == -3);
    data[0] = 1;
    data[1] = 2;
    data[2] = 3;
  }

  //The test behavior only allows 5 steps
  REQUIRE(!behav->canStep());
}

TEST_CASE("loading and releasing", "[PyLoadableBehavior]") {
  bl_loader::BLLoader loader;
  REQUIRE_NOTHROW(loader.acquireBehavior("TestBehavior"));
  REQUIRE_NOTHROW(loader.releaseLibrary("TestBehavior"));
}

TEST_CASE("acquire twice", "[PyLoadableBehavior]") {
  bl_loader::BLLoader loader;
  LoadableBehavior* b = loader.acquireBehavior("TestBehavior");
  LoadableBehavior* b2 = loader.acquireBehavior("TestBehavior");
  REQUIRE(b == b2);
}


