#include "PyEnvironment.h"
#include <limits>
#include <algorithm>
#include <cassert>
#include <stdexcept>


namespace bolero { namespace bl_loader {

PyEnvironment::PyEnvironment(lib_manager::LibManager *theManager,
		                     const std::string libName, int libVersion)
  : bolero::Environment(theManager, libName, libVersion)
{
    environment = PythonInterpreter::instance()
        .import("bolero.utils.module_loader")
        ->function("environment_from_yaml").call()
        .returnObject();
    if(!environment)
        std::runtime_error("Environment construction failed");
}

void PyEnvironment::init(std::string config) {
  environment->method("init").call();
}

void PyEnvironment::reset() {
  environment->method("reset").call();
}

int PyEnvironment::getNumInputs() const {
  return environment->method("get_num_inputs").call().returnObject()->asInt();
}

int PyEnvironment::getNumOutputs() const {
  return environment->method("get_num_outputs").call().returnObject()->asInt();
}

void PyEnvironment::getOutputs(double *values, int numOutputs) const {
  environment->method("get_outputs").pass(ONEDCARRAY).call(values, numOutputs);
}

void PyEnvironment::setInputs(const double *values, int numInputs) {
  environment->method("set_inputs").pass(ONEDCARRAY).call(values, numInputs);
}

int PyEnvironment::getFeedback(double *feedback) const {
  shared_ptr<Object> result = environment->method("get_feedback")
    .call().returnObject();
  shared_ptr<std::vector<double> > feedbackVector = result->as1dArray();

  const int numFeedbacks = (int) feedbackVector->size();
  std::copy(feedbackVector->begin(), feedbackVector->end(), feedback);
  return numFeedbacks;
}

void PyEnvironment::stepAction() {
  environment->method("step_action").call();
}

bool PyEnvironment::isEvaluationDone() const {
  return environment->method("is_evaluation_done")
    .call().returnObject()->asBool();
}

bool PyEnvironment::isBehaviorLearningDone() const {
  return environment->method("is_behavior_learning_done")
    .call().returnObject()->asBool();
}

}}
