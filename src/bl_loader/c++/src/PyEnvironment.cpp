#include "PyEnvironment.h"
#include <limits>
#include <cassert>

namespace bolero { namespace bl_loader {

PyEnvironment::PyEnvironment(lib_manager::LibManager *theManager,
		                     const std::string libName, int libVersion)
  : bolero::Environment(theManager, libName, libVersion)
{
    environment = PythonInterpreter::instance()
        .import("bolero.utils.module_loader")
        ->function("environment_from_yaml").call()
        .returnObject();
}

void PyEnvironment::init() {
  environment->method("init").call();
}

void PyEnvironment::reset() {
  environment->method("reset").call();
}

int PyEnvironment::getNumInputs() const {
  return environment->method("get_num_inputs").call().returnObject().asInt();
}

int PyEnvironment::getNumOutputs() const {
  return environment->method("get_num_outputs").call().returnObject().asInt();
}

void PyEnvironment::getOutputs(double *values, int numOutputs) const {
  // TODO
  environment->method("get_outputs").pass(ONEDARRAY).call(array);
}

void PyEnvironment::setInputs(const double *values, int numInputs) {
  // TODO
  environment->method("set_inputs").pass(ONEDARRAY).call(array);
}

int PyEnvironment::getFeedback(double *feedback) const {
  std::vector<double> feedback = environment.method("get_feedback").call().returnObject().as1dArray();
  // TODO assign feedback
  return feedback.size();
}

void PyEnvironment::stepAction() {
  environment->method("step_action").call();
}

bool PyEnvironment::isEvaluationDone() const {
  return environment->method("is_evaluation_done").call().returnObject().asBool();
}

bool PyEnvironment::isBehaviorLearningDone() const {
  return environment->method("is_behavior_learning_done").call().returnObject().asBool();
}

}}
