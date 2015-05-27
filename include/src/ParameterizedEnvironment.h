/**
 * \file ParameterizedEnvironment.h
 * \author Constantin Bergatt
 * \brief A interface for a parameterized environment for the machine learning framework.
 *
 * Version 0.1
 */

#ifndef BL_FRAMEWORK_PARAMETERIZED_ENVIRONMENT_H
#define BL_FRAMEWORK_PARAMETERIZED_ENVIRONMENT_H

#ifdef _PRINT_HEADER_
  #warning "ParameterizedEnvironment.h"
#endif

#include <Environment.h>

namespace bolero {

  class ParameterizedEnvironment : public virtual Environment {

  public:
    ParameterizedEnvironment(lib_manager::LibManager *theManager,
                             const std::string &libName, int libVersion)
      : Environment(theManager, libName, libVersion) {
    }

    virtual ~ParameterizedEnvironment() {}

    virtual int getNumParameters() const = 0;
    virtual void getParameters(double *values, int numParameters) const = 0;
    virtual void setParameters(const double *values, int numParameters) = 0;

  }; // end of class definition ParameterizedEnvironment

} // end of namespace bolero

#endif // BL_FRAMEWORK_PARAMETERIZED_ENVIRONMENT_H
