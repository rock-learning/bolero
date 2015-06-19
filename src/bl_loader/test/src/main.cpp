#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include <LoadableBehavior.h>
#include <BLLoader.h>
#include <Optimizer.h>
#include <Environment.h>
#include <BehaviorSearch.h>
#include <PythonInterpreter.hpp>

using namespace bolero;
using namespace bolero::bl_loader;
using namespace std;


TEST_CASE( "io", "[PythonInterpreter]" ) {
  const PythonInterpreter& python = PythonInterpreter::instance();
  shared_ptr<Module> functions = python.import("functions");
  int intResult = functions->function("produce_int").call().returnObject()->asInt();
  REQUIRE(intResult == 9);
  double doubleResult = functions->function("produce_double").call().returnObject()->asDouble();
  REQUIRE(doubleResult == 8.0);
  bool boolResult = functions->function("produce_bool").call().returnObject()->asBool();
  REQUIRE(boolResult == true);
  std::string stringResult = functions->function("produce_string").call().returnObject()->asString();
  REQUIRE(stringResult == "Test string");

  functions->function("take_int").pass(INT).call(intResult);
  functions->function("take_double").pass(DOUBLE).call(doubleResult);
  functions->function("take_bool").pass(BOOL).call(boolResult);
  functions->function("take_string").pass(STRING).call(&stringResult);

  shared_ptr<ListBuilder> list = python.listBuilder();
  list->pass(DOUBLE).build(1.0);
  list->pass(DOUBLE).build(2.0);
  list->pass(DOUBLE).build(3.0);
  shared_ptr<std::vector<double> > vector = list->build()->as1dArray();
  REQUIRE(vector->at(0) == 1.0);
  REQUIRE(vector->at(1) == 2.0);
  REQUIRE(vector->at(2) == 3.0);
}

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
    behav->setInputs(data, 3);
    behav->step();
    behav->getOutputs(data, 3);

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

TEST_CASE( "optimize", "[PyOptimizer]" ) {
  // Load optimizer that is defined by "learning_config.yml"
  bl_loader::BLLoader loader;
  Optimizer* opt = loader.acquireOptimizer("Python");

  int n_params = 3;
  double params[n_params];
  int n_feedbacks = 2;
  double feedback[n_feedbacks];

  // Required step: initialize optimizer
  opt->init(n_params);

  for(int i = 0; i < 200; i++)
  {
    REQUIRE(!opt->isBehaviorLearningDone());
    // Required step: generate next possible solution
    opt->getNextParameters(params, n_params);

    // Compute feedback
    feedback[0] = 10.0 - (params[0] + params[1]);
    feedback[0] *= -feedback[0];
    feedback[1] = 5.0 - params[2];
    feedback[1] *= -feedback[1];

    // Required step: tell the optimizer the performance of the solution
    opt->setEvaluationFeedback(feedback, n_feedbacks);
  }
}

TEST_CASE( "environment", "[PyEnvironment]" ) {
  // Load environment that is defined by "learning_config.yml"
  bl_loader::BLLoader loader;
  Environment* env = loader.acquireEnvironment("Python");
  REQUIRE_NOTHROW(env->init());
  REQUIRE_NOTHROW(env->reset());
  REQUIRE(!env->isEvaluationDone());
  REQUIRE(env->getNumInputs() == 3);
  REQUIRE(env->getNumOutputs() == 0);
  double params[3] = {0.0, 0.0, 0.0};
  REQUIRE_NOTHROW(env->setInputs(params, 3));
  REQUIRE_NOTHROW(env->stepAction());
  double result[0];
  REQUIRE_NOTHROW(env->getOutputs(result, 0));
  REQUIRE(env->isEvaluationDone());
  double feedback[1];
  const int numFeedbacks = env->getFeedback(feedback);
  REQUIRE(numFeedbacks == 1);
  REQUIRE(Approx(-589729.9344730391) == feedback[0]);
  REQUIRE(!env->isBehaviorLearningDone());
}

TEST_CASE( "behavior_search", "[PyBehaviorSearch]" ) {
  // Load behavior search that is defined by "learning_config.yml"
  bl_loader::BLLoader loader;
  BehaviorSearch* bs = loader.acquireBehaviorSearch("Python");
  REQUIRE_NOTHROW(bs->init(3, 0));
  Behavior* beh = bs->getNextBehavior();
  double outputs[3];
  REQUIRE_NOTHROW(beh->getOutputs(outputs, 3));
  REQUIRE(outputs[0] == Approx(1.764052346));
  REQUIRE(outputs[1] == Approx(0.4001572084));
  REQUIRE(outputs[2] == Approx(0.9787379841));
  double feedback[1] = {0.0};
  bs->setEvaluationFeedback(feedback, 1);
  beh = bs->getBestBehavior();
  REQUIRE_NOTHROW(beh->getOutputs(outputs, 3));
  REQUIRE(outputs[0] == Approx(1.764052346));
  REQUIRE(outputs[1] == Approx(0.4001572084));
  REQUIRE(outputs[2] == Approx(0.9787379841));
  REQUIRE(!bs->isBehaviorLearningDone());
}
