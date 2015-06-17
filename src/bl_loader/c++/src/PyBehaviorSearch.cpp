#include <Python.h>
#include "PyBehaviorSearch.h"
#include "PyBehavior.h"
#include <cassert>

namespace bolero { namespace bl_loader {

PyBehaviorSearch::PyBehaviorSearch(lib_manager::LibManager *theManager,
                                   const std::string libName, int libVersion)
  : bolero::BehaviorSearch(theManager, libName, libVersion)
{
    behaviorSearch = PythonInterpreter::instance()
        .import("bolero.utils.module_loader")
        ->function("behavior_search_from_yaml").call()
        .returnObject();
}

void PyBehaviorSearch::init(int numInputs, int numOutputs) {
  behaviorSearch->method("init").pass(INT).pass(INT).call(numInputs, numOutputs);
}

bolero::Behavior* PyBehaviorSearch::getNextBehavior() {
  shared_ptr<Object> behavior = behaviorSearch
    .method("get_next_behavior").call().returnObject();
  bolero::Behavior* ret = PyBehavior::fromPyObject(pResult); // TODO
  return ret;
}

void PyBehaviorSearch::setEvaluationFeedback(const double *feedbacks,
                                             int numFeedbacks) {
  behaviorSearch->method("set_evaluation_feedback").pass(ONEDARRAY).call(&array); // TODO
}

void PyBehaviorSearch::writeResults(const std::string &resultPath) {
  std::string path = resultPath;
  behaviorSearch->method("write_results").pass(STRING).call(&path);
}

bolero::Behavior* PyBehaviorSearch::getBehaviorFromResults(const std::string &resultPath) {
  std::string path = resultPath;
  shared_ptr<Object> behavior = behaviorSearch
    .method("get_behavior_from_results").pass(STRING).call(&path)
    .returnObject();
  bolero::Behavior* ret = PyBehavior::fromPyObject(pResult); // TODO
}

bool PyBehaviorSearch::isBehaviorLearningDone() const {
  return behaviorSearch->method("is_behavior_learning_done").call().returnObject().asBool();
}

}}
