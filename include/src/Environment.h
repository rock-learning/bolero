/**
 * \file Environment.h
 * \author Malte Langosz
 * \brief A interface for a environment for the machine learning framework.
 *
 * Version 0.1
 */

#ifndef BL_FRAMEWORK_ENVIRONMENT_H
#define BL_FRAMEWORK_ENVIRONMENT_H

#ifdef _PRINT_HEADER_
  #warning "Environment.h"
#endif

#include <lib_manager/LibInterface.hpp>
#include <string>

namespace bolero {

  class Behavior;

  class Environment : public lib_manager::LibInterface {

  public:
    Environment(lib_manager::LibManager *theManager,
                const std::string &libName, int libVersion) :
      lib_manager::LibInterface(theManager), libName(libName),
      libVersion(libVersion) {
    }

    virtual ~Environment() {}

    // LibInterface methods
    virtual int getLibVersion() const {return libVersion;}
    virtual const std::string getLibName() const {return libName;}
    virtual void createModuleInfo() {}

    // Environment methods
    virtual void init() = 0;
    virtual void reset() = 0;

    /**
     * This functions are used for the controller interfacing a
     * behavior to an environment
     */
    virtual int getNumInputs() const = 0;
    virtual int getNumOutputs() const = 0;
    virtual void getOutputs(double *values, int numOutputs) const = 0;
    virtual void setInputs(const double *values, int numInputs) = 0;
    virtual void stepAction() = 0;

    // sets whether the environment is in training or test mode
    virtual void setTestMode(bool b) {}

    virtual bool isEvaluationDone() const = 0;

    // returns how many rewards were assigned to the pointer parameter
    // for the whole evaluation
    virtual int getFeedback(double *feedback) const = 0;

    virtual bool isBehaviorLearningDone() const = 0;

  protected:
    std::string libName;
    int libVersion;

  }; // end of class definition Environment

} // end of namespace bolero

#endif // BL_FRAMEWORK_ENVIRONMENT_H
