#include <Python.h>
#include "PyEnvironment.h"
#include <limits>
#include <cassert>

namespace behavior_learning { namespace bl_loader {

PyEnvironment::PyEnvironment(lib_manager::LibManager *theManager,
		const std::string libName, int libVersion)
  : behavior_learning::Environment(theManager, libName, libVersion),
    py_environment(Helper::instance().getClassInstance("environment"))
{}

PyEnvironment::~PyEnvironment() {

  Helper::instance().destroyCallableInfo(&get_outputs);
  Helper::instance().destroyCallableInfo(&set_inputs);
  Helper::instance().destroyCallableInfo(&get_step_feedback);
  Helper::instance().destroyCallableInfo(&get_feedbacks);
}


void PyEnvironment::init() {
  assert(py_environment);
  PyObject *pResult;
  pResult = PyObject_CallMethod(py_environment.get(), (char*)"init", NULL);
  if(PyErr_Occurred()) {
    Helper::instance().printPyTraceback();
  }
  size_t numInputs = getNumInputs();
  size_t numOutputs = getNumOutputs();
  Helper::instance().createCallableInfo(&get_outputs, py_environment.get(), "get_outputs", numOutputs);
  Helper::instance().createCallableInfo(&set_inputs, py_environment.get(), "set_inputs", numInputs);
  Helper::instance().createCallableInfo(&get_step_feedback, py_environment.get(),
                     "get_step_feedback", 1);
  Py_XDECREF(pResult);
}

void PyEnvironment::reset() {
  assert(py_environment);
  PyObject *pResult;
  pResult = PyObject_CallMethod(py_environment.get(), (char*)"reset", NULL);
  if(PyErr_Occurred()) {
    PyErr_Print();
  }
  Py_XDECREF(pResult);
}

int PyEnvironment::getNumInputs() const {
  assert(py_environment);
  long ret = -1;
  PyObject *pResult;
  pResult = PyObject_CallMethod(py_environment.get(), (char*)"get_num_inputs", NULL);
  if(pResult) {
    assert(PyInt_Check(pResult));
    ret = PyInt_AsLong(pResult);
    Py_DECREF(pResult);
    if(ret < 0 || ret > std::numeric_limits<int>::max()) {
      fprintf(stderr, "getNumInputs created an overflow!\n");
      ret = -1;
    }
  }
  if(PyErr_Occurred()) {
    PyErr_Print();
  }
  return ret;
}

int PyEnvironment::getNumOutputs() const {
  assert(py_environment);
  long ret = -1;
  PyObject *pResult;
  pResult = PyObject_CallMethod(py_environment.get(), (char*)"get_num_outputs", NULL);
  if(pResult) {
    assert(PyInt_Check(pResult));
    ret = PyInt_AsLong(pResult);
    Py_DECREF(pResult);
    if(ret < 0 || ret > std::numeric_limits<int>::max()) {
      fprintf(stderr, "getNumOutputs created an overflow!\n");
      ret = -1;
    }
  }
  if(PyErr_Occurred()) {
    PyErr_Print();
  }
  return ret;
}

#ifdef USE_MEMORYVIEWS

void PyEnvironment::getOutputs(double *values, int numOutputs) const {
  assert(py_environment);
  PyObjectPtr memView = Helper::instance().create1dBuffer(values, numOutputs);
  PyObject *result = PyObject_CallMethod(py_environment.get(), (char*)"get_outputs",
                                         (char*)"O", memView.get());
  Py_XDECREF(result);

  if(PyErr_Occurred()) {
    PyErr_Print();
  }
}

void PyEnvironment::setInputs(const double *values, int numInputs){
  assert(py_environment);
  PyObjectPtr memView = Helper::instance().create1dBuffer(values, numInputs);

  PyObject *result = PyObject_CallMethod(py_environment.get(), (char*)"set_inputs",
                                         (char*)"O", memView.get());
  Py_XDECREF(result);

  if(PyErr_Occurred()){
    PyErr_Print();
  }
}

#else // USE_MEMORYVIEWS

void PyEnvironment::getOutputs(double *values, int numOutputs) const {
  assert(py_environment);
  PyObject *result = PyObject_CallObject(get_outputs.callable,
                                         get_outputs.argTuple);
  if(result) {
    Helper::instance().extractFromCallableInfo(&get_outputs, values, numOutputs);
    Py_DECREF(result);
  } else {
    PyErr_Print();
  }
}

void PyEnvironment::setInputs(const double *values, int numInputs){
  assert(py_environment);
  Helper::instance().fillCallableInfo(&set_inputs, values, numInputs);
  PyObject *result = PyObject_CallObject(set_inputs.callable,
                                         set_inputs.argTuple);
  if(result) {
    Py_DECREF(result);
  } else {
    PyErr_Print();
  }
}

#endif // USE_MEMORYVIEWS

int PyEnvironment::getFeedback(double *feedback) const {
  assert(py_environment);
  PyObjectPtr memView = Helper::instance().create1dBuffer(feedback, 1000);
  long ret = -1;

  PyObject *result = PyObject_CallMethod(py_environment.get(), (char*)"get_feedback",
                                         (char*)"O", memView.get());
  if(result) {
    assert(PyInt_Check(result));
    ret = PyInt_AsLong(result);
  }
  Py_XDECREF(result);

  if(PyErr_Occurred()){
    PyErr_Print();
  }
  return ret;
}

void PyEnvironment::stepAction() {
  assert(py_environment);
  PyObject *result = PyObject_CallMethod(py_environment.get(), (char*)"step_action",
                                         NULL);
  if(!result) {
    PyErr_Print();
  }
  Py_XDECREF(result);
}

bool PyEnvironment::isEvaluationDone() const {
  assert(py_environment);
  bool ret = true;
  PyObject *result;
  result = PyObject_CallMethod(py_environment.get(), (char*)"is_evaluation_done",
                               NULL);
  if(result) {
    assert(PyBool_Check(result));
    ret = PyObject_IsTrue(result);
    Py_DECREF(result);
  } else {
    PyErr_Print();
  }
  return ret;
}

bool PyEnvironment::isBehaviorLearningDone() const {
  assert(py_environment);
  bool ret = false;
  PyObject *result;
  result = PyObject_CallMethod(py_environment.get(),
                               (char*)"is_behavior_learning_done", NULL);
  if(result) {
    assert(PyBool_Check(result));
    ret = PyObject_IsTrue(result);
    Py_DECREF(result);
  } else {
    PyErr_Print();
  }
  return ret;
}

}}