/**
 * \file ContextualEnvironment.h
 * \author Constantin Bergatt
 * \brief A interface for a contextual environment for the machine learning framework.
 *
 * Version 0.1
 */

#ifndef BL_FRAMEWORK_CONTEXTUAL_ENVIRONMENT_H
#define BL_FRAMEWORK_CONTEXTUAL_ENVIRONMENT_H

#ifdef _PRINT_HEADER_
  #warning "ContextualEnvironment.h"
#endif

#include <Environment.h>

namespace bolero {

  class ContextualEnvironment : public virtual Environment {

  public:
    ContextualEnvironment(lib_manager::LibManager *theManager,
                          const std::string &libName, int libVersion)
      : Environment(theManager, libName, libVersion) {
    }

    virtual ~ContextualEnvironment() {}

    virtual double* request_context(double *context, int numContext) = 0;
    virtual int get_num_context_dims() const = 0;
    virtual bool isContextual() {return true;};

  }; // end of class definition ContextualEnvironment

} // end of namespace bolero

#endif // BL_FRAMEWORK_CONTEXTUAL_ENVIRONMENT_H
