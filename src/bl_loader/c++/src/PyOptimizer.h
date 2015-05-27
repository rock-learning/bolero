/**
 * \file Optimizer.h
 * \author Alexander Fabisch, Lorenz Quack
 * \brief A generic wrapper around a parameter optimization written in Python.
 */

#ifndef BL_OPTIMIZER_H
#define BL_OPTIMIZER_H

#ifdef _PRINT_HEADER_
  #warning "PyOptimizer.h"
#endif

#include "Helper.h"
#include <string>
#include <Optimizer.h>
#include "PyLoadable.h"

// forward declare PyObject
// as suggested on the python mailing list
// http://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif


namespace behavior_learning { namespace bl_loader {

  class PyOptimizer : public Optimizer, public PyLoadable {
  public:
    PyOptimizer(lib_manager::LibManager *theManager,
    		std::string libName, int libVersion);
    virtual ~PyOptimizer();

    // Optimizer methods
    virtual void init(int dimension);
    virtual void getNextParameters(double *p, int numP);
    virtual void getBestParameters(double *p, int numP);
    virtual void setEvaluationFeedback(const double *feedbacks,
    		int numFeedbacks);
    virtual bool isBehaviorLearningDone() const;
    virtual std::vector<double*> getNextParameterSet() const;
    virtual void setParameterSetFeedback(const std::vector<double> feedback);

  private:
    PyObjectPtr py_optimizer;
    size_t dimension;
    py_callable_info_t get_next_parameters;
    py_callable_info_t get_best_parameters;
    py_callable_info_t set_evaluation_feedback;

  }; // end of class definition Py_Optimizer

}} //end namespaces

#endif // BL_OPTIMIZER_H
