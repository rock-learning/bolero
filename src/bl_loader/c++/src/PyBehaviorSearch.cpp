#include <Python.h>
#include "PyBehaviorSearch.h"
#include "PyBehavior.h"
#include <cassert>

namespace bolero { namespace bl_loader {

PyBehaviorSearch::PyBehaviorSearch(lib_manager::LibManager *theManager,
                                   const std::string libName, int libVersion)
  : bolero::BehaviorSearch(theManager, libName, libVersion), behavior(0),
    bestBehavior(0)
{
    behaviorSearch = PythonInterpreter::instance()
      .import("bolero.utils.module_loader")
      ->function("behavior_search_from_yaml").call()
      .returnObject();
}

PyBehaviorSearch::~PyBehaviorSearch()
{
  if(behavior)
    delete behavior;
  if(bestBehavior)
    delete bestBehavior;
}

void PyBehaviorSearch::init(int numInputs, int numOutputs) {
  behaviorSearch->method("init")
    .pass(INT).pass(INT).call(numInputs, numOutputs);
}

bolero::Behavior* PyBehaviorSearch::getNextBehavior() {
  if(behavior)
    delete behavior;
  shared_ptr<Object> behaviorObject = behaviorSearch
    ->method("get_next_behavior").call().returnObject();
  behavior = PyBehavior::fromPyObject(behaviorObject);
  return behavior;
}

bolero::Behavior* PyBehaviorSearch::getBestBehavior() {
  if(bestBehavior)
    delete bestBehavior;
  shared_ptr<Object> behaviorObject = behaviorSearch
    ->method("get_best_behavior").call().returnObject();
  bestBehavior = PyBehavior::fromPyObject(behaviorObject);
  return bestBehavior;
}

void PyBehaviorSearch::setEvaluationFeedback(
    const double *feedbacks, int numFeedbacks) {
  behaviorSearch->method("set_evaluation_feedback")
    .pass(ONEDCARRAY).call(feedbacks, numFeedbacks);
}

void PyBehaviorSearch::writeResults(const std::string &resultPath) {
  std::string path = resultPath;
  behaviorSearch->method("write_results").pass(STRING).call(&path);
}

bolero::Behavior* PyBehaviorSearch::getBehaviorFromResults(
    const std::string &resultPath) {
  if(behavior)
    delete behavior;
  std::string path = resultPath;
  shared_ptr<Object> behaviorObject = behaviorSearch
    ->method("get_behavior_from_results").pass(STRING).call(&path)
    .returnObject();
  behavior = PyBehavior::fromPyObject(behaviorObject);
  return behavior;
}

bool PyBehaviorSearch::isBehaviorLearningDone() const {
  return behaviorSearch->method("is_behavior_learning_done").call()
    .returnObject()->asBool();
}

}}
