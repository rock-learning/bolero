#include <Python.h>
#include "PyBehaviorSearch.h"
#include "PyBehavior.h"
#include <cassert>

namespace bolero { namespace bl_loader {

PyBehaviorSearch::PyBehaviorSearch(lib_manager::LibManager *theManager,
                                   const std::string libName, int libVersion)
  : bolero::BehaviorSearch(theManager, libName, libVersion),
    py_behaviorSearch(Helper::instance().getClassInstance("behavior_search"))
{}

PyBehaviorSearch::~PyBehaviorSearch() {
  Helper::instance().destroyCallableInfo(&set_evaluation_feedback);
}


void PyBehaviorSearch::init(int numInputs, int numOutputs) {
  assert(py_behaviorSearch);
  PyObject *result;
  result = PyObject_CallMethod(
    py_behaviorSearch.get(), (char*)"init", (char*)"ii", numInputs, numOutputs);
  if(!result) {
    PyErr_Print();
  }
  Helper::instance().createCallableInfo(
    &set_evaluation_feedback, py_behaviorSearch.get(),
    "set_evaluation_feedback", 1);
  Py_XDECREF(result);
}

bolero::Behavior* PyBehaviorSearch::getNextBehavior() {
  assert(py_behaviorSearch);
  bolero::Behavior *ret = NULL;
  PyObject *pResult;
  pResult = PyObject_CallMethod(
    py_behaviorSearch.get(), (char*)"get_next_behavior", NULL);
  if(pResult) {
    ret = PyBehavior::fromPyObject(pResult);
    Py_DECREF(pResult);
  } else {
    PyErr_Print();
  }
  return ret;
}

#ifdef USE_MEMORYVIEWS

void PyBehaviorSearch::setEvaluationFeedback(const double *feedbacks,
                                             int numFeedbacks) {
  assert(py_behaviorSearch);
  PyObjectPtr memView = Helper::instance().create1dBuffer(feedbacks, numFeedbacks);

  PyObject *pResult = PyObject_CallMethod(
    py_behaviorSearch.get(), (char*)"set_evaluation_feedback",
    (char*)"O", memView.get());
  Py_XDECREF(pResult);

  if(PyErr_Occurred()) {
    PyErr_Print();
  }
}

#else // USE_MEMORYVIEWS

void PyBehaviorSearch::setEvaluationFeedback(const double *feedbacks,
                                             int numFeedbacks) {
  Helper::instance().fillCallableInfo(
    &set_evaluation_feedback, feedbacks, numFeedbacks);
  PyObject *pResult = PyObject_CallObject(
    set_evaluation_feedback.callable, set_evaluation_feedback.argTuple);
  if(pResult) {
    Py_DECREF(pResult);
  } else {
    PyErr_Print();
  }
}

#endif // USE_MEMORYVIEWS

void PyBehaviorSearch::writeResults(const std::string &resultPath) {
  assert(py_behaviorSearch);
  PyObject *pResult;
  pResult = PyObject_CallMethod(
    py_behaviorSearch.get(), (char*)"write_results", (char*)"s",
    resultPath.c_str());
  if(!pResult) {
    PyErr_Print();
  }
  Py_XDECREF(pResult);
}

bolero::Behavior* PyBehaviorSearch::getBehaviorFromResults(const std::string &resultPath) {
  assert(py_behaviorSearch);
  bolero::Behavior *ret = NULL;
  PyObject *pResult = PyObject_CallMethod(
    py_behaviorSearch.get(), (char*)"getBehaviorFromResults",
    (char*)"s", resultPath.c_str());
  if(pResult) {
    ret = PyBehavior::fromPyObject(pResult);
    Py_DECREF(pResult);
  } else {
    PyErr_Print();
  }
  return ret;
}

bool PyBehaviorSearch::isBehaviorLearningDone() const {
  assert(py_behaviorSearch);
  PyObject *result;
  bool ret = false;
  result = PyObject_CallMethod(
    py_behaviorSearch.get(), (char*)"is_behavior_learning_done", NULL);
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
