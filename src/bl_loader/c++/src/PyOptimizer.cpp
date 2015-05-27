#include <Python.h>
#include "PyOptimizer.h"
#include "Helper.h"
#include <cassert>


namespace behavior_learning { namespace bl_loader {

  PyOptimizer::PyOptimizer(lib_manager::LibManager *theManager,
                             std::string libName, int libVersion)
    : behavior_learning::Optimizer(theManager, libName, libVersion),
      py_optimizer(Helper::instance().getClassInstance("optimizer")),
      dimension(0)
  {}

  PyOptimizer::~PyOptimizer() {
    Helper::instance().destroyCallableInfo(&get_next_parameters);
    Helper::instance().destroyCallableInfo(&get_best_parameters);
    Helper::instance().destroyCallableInfo(&set_evaluation_feedback);
  }

  void PyOptimizer::init(int dimension) {
    assert(py_optimizer);
    PyObject *pResult = PyObject_CallMethod(py_optimizer.get(),(char*)"init",
                                            (char*)"i", dimension);
    if(!pResult) {
      PyErr_Print();
    }
    Py_XDECREF(pResult);
    Helper::instance().createCallableInfo(&get_next_parameters, py_optimizer.get(),
                       "get_next_parameters", dimension);
    Helper::instance().createCallableInfo(&get_best_parameters, py_optimizer.get(),
                       "get_best_parameters", dimension);
    Helper::instance().createCallableInfo(&set_evaluation_feedback, py_optimizer.get(),
                       "set_evaluation_feedback", 1);
    this->dimension = dimension;
  }

  void PyOptimizer::getNextParameters(double *p, int numP) {
    assert(py_optimizer);
    PyObjectPtr memView = Helper::instance().create1dBuffer(p, numP);

    PyObject *pResult = PyObject_CallMethod(
        py_optimizer.get(),
        (char*)"get_next_parameters",
        (char*)"O",
        memView.get());
    Py_XDECREF(pResult);

    if(PyErr_Occurred()) {
      PyErr_Print();
    }
  }

  void PyOptimizer::getBestParameters(double *p, int numP) {
    assert(py_optimizer);
    PyObjectPtr memView = Helper::instance().create1dBuffer(p, numP);

    PyObject *pResult = PyObject_CallMethod(
        py_optimizer.get(),
        (char*)"get_best_parameters",
        (char*)"O",
        memView.get());
    Py_XDECREF(pResult);

    if(PyErr_Occurred()) {
      PyErr_Print();
    }
  }

  void PyOptimizer::setEvaluationFeedback(const double *feedbacks,
                                           int numFeedbacks) {
    assert(py_optimizer);

    PyObjectPtr memView = Helper::instance().create1dBuffer(feedbacks, numFeedbacks);

    PyObject *pResult = PyObject_CallMethod(
        py_optimizer.get(),
        (char*)"set_evaluation_feedback",
        (char*)"O",
        memView.get());
    Py_XDECREF(pResult);

    if(PyErr_Occurred()) {
      PyErr_Print();
    }
  }

  bool PyOptimizer::isBehaviorLearningDone() const {
    assert(py_optimizer);
    PyObject *result;
    bool ret = false;
    result = PyObject_CallMethod(py_optimizer.get(),
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

  std::vector<double*> PyOptimizer::getNextParameterSet() const {
    fprintf(stderr, "PyOptimizer::getNextParameterSet() not implemented yet!\n");
    assert(false);
  }

  void PyOptimizer::setParameterSetFeedback(const std::vector<double> feedback) {
	fprintf(stderr, "PyOptimizer::setParameterSetFeedback() not implemented yet!\n");
    assert(false);
  }


}}
