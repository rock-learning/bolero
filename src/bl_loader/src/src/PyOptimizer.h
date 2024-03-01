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

#include <PythonInterpreter.hpp>
#include <string>
#include <Optimizer.h>
#include "PyLoadable.h"


namespace bolero { namespace bl_loader {

  class PyOptimizer : public Optimizer, public PyLoadable {
  public:
    PyOptimizer(lib_manager::LibManager *theManager, std::string libName,
                int libVersion);

    // Optimizer methods
    virtual void init(int dimension, std::string config="");
    virtual void getNextParameters(double *p, int numP);
    virtual void getBestParameters(double *p, int numP);
    virtual void setEvaluationFeedback(const double *feedbacks,
                                       int numFeedbacks);
    virtual bool isBehaviorLearningDone() const;
    virtual void getNextParameterSet(double *p, int numP, int batchSize) const;
    virtual void setParameterSetFeedback(const double *feedback, int numFeedbacksPerSet, int batchSize);
    virtual int getBatchSize() const;

  private:
    shared_ptr<Object> optimizer;
    size_t dimension;
  }; // end of class definition Py_Optimizer

}} //end namespaces

#endif // BL_OPTIMIZER_H
