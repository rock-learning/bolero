#include <Python.h>
#include "PyEnvironment.h"
#include <limits>
#include <algorithm>
#include <cassert>
#include <stdexcept>


namespace bolero { namespace bl_loader {

PyEnvironment::PyEnvironment(lib_manager::LibManager *theManager,
                         const std::string libName, int libVersion)
  : bolero::Environment(theManager, libName, libVersion), environment(0)
{
}

void PyEnvironment::init(std::string config) {
  if(config == "")
    config = "Environment:\n    type: " + libName;
  environment = PythonInterpreter::instance()
    .import("bolero.utils.module_loader")
    ->function("environment_from_yaml_string").pass(STRING).call(&config)
    .returnObject();
  if(!environment)
    std::runtime_error("Environment construction failed");
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

void PyEnvironment::setBehavior(Behavior *behavior) {
/*  - we create a CppBehavior wrapper
 *  - set the thisptr attribute to the cpp behavior ptr
 *  - todo: check if this is working
 *  - call the set_behavior method with the newly created CppBehavior object
 */
  PyObject *pyWrapper = PyImport_ImportModule("bolero.wrapper");
  PyObject *pyBehavior = PyObject_GetAttrString(pyWrapper, "CppBehavior");
  Py_DECREF(pyWrapper);
  PyObject_SetAttrString(pyBehavior, "thisptr", (PyObject*)behavior);
  environment->method("set_behavior").pass(OBJECT).call(pyBehavior);
  Py_DECREF(pyBehavior);
}

int PyEnvironment::getFeedback(double *feedback) const {
  shared_ptr<Object> result = environment->method("get_feedback")
    .call().returnObject();
  shared_ptr<std::vector<double> > feedbackVector = result->as1dArray();

  const int numFeedbacks = (int) feedbackVector->size();
  std::copy(feedbackVector->begin(), feedbackVector->end(), feedback);
  return numFeedbacks;
}

int PyEnvironment::getStepFeedback(double *feedback) const {
  shared_ptr<Object> result = environment->method("get_step_feedback")
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

bool PyEnvironment::isEvaluationAborted() const {
  return environment->method("is_evaluation_aborted")
    .call().returnObject()->asBool();
}

bool PyEnvironment::isBehaviorLearningDone() const {
  return environment->method("is_behavior_learning_done")
    .call().returnObject()->asBool();
}

}}
