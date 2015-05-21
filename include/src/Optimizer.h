/**
 * \file Optimizer.h
 * \author Malte Langosz
 * \brief A interface for parameter optimization algorithms.
 *
 * Version 0.1
 */

#ifndef __BL_OPTIMIZER_H__
#define __BL_OPTIMIZER_H__

#ifdef _PRINT_HEADER_
  #warning "Optimizer.h"
#endif

#include <string>
#include <vector>

#include <lib_manager/LibInterface.hpp>

namespace behavior_learning {

  class Optimizer : public lib_manager::LibInterface {

  public:
    Optimizer(lib_manager::LibManager *theManager,
              std::string libName, int libVersion)
      : lib_manager::LibInterface(theManager),
        libName(libName),
        libVersion(libVersion) {
    }

    virtual ~Optimizer() {}

    // LibInterface methods
    int getLibVersion() const {return libVersion;}
    const std::string getLibName() const {return libName;}
    virtual void createModuleInfo() {}

    // Optimizer methods
    virtual void init(int dimension) = 0;
    virtual void getNextParameters(double *p, int numP) = 0;
    virtual void getBestParameters(double *p, int numP) = 0;
    virtual void setEvaluationFeedback(const double *feedbacks,
                                       int numFeedbacks) = 0;
    virtual bool isBehaviorLearningDone() const = 0;

    virtual std::vector<double*> getNextParameterSet() const = 0;
    virtual void setParameterSetFeedback(const std::vector<double> feedback) = 0;

  protected:
    int dimension;
    std::string libName;
    int libVersion;
  }; // end of class definition Optimizer

} // end of namespace behavior_learning

#endif // __BL_OPTIMIZER_H__
