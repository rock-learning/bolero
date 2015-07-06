/**
 * \file MARSEnvironmentHelper.h
 * \author Malte Langosz
 * \brief 
 *
 * Version 0.1
 */

#ifndef __MARS_ENVIRONMENT_HELPER_H__
#define __MARS_ENVIRONMENT_HELPER_H__

#ifdef _PRINT_HEADER_
  #warning "MARSEnvironmentHelper.h"
#endif

#include <Environment.h>
#include <cassert>

#include "MARSEnvPlugin.h"


namespace bolero {
  namespace mars_environment {

    class MARSThread;

    class MARSEnvironmentHelper : public virtual Environment {

    public:
      MARSEnvironmentHelper(lib_manager::LibManager *theManager,
                            const char* name, int version,
                            MARSEnvPlugin *marsPlugin);
      virtual ~MARSEnvironmentHelper();

      CREATE_MODULE_INFO();

      virtual void init();
      virtual void reset();

      /**
       * This functions are used for the controller interfacing a
       * behavior to an environment
       */
      virtual int getNumOutputs() const = 0;
      virtual int getNumInputs() const = 0;

      virtual void getOutputs(double *values, int numOutputs) const;
      virtual void setInputs(const double *values, int numInputs);
      virtual void stepAction();

      virtual bool isBehaviorLearningDone() const = 0;

    private:
      MARSThread *marsThread;
      MARSEnvPlugin *marsPlugin;
      unsigned int graphicsUpdateTime;

    }; // end of class definition MARSEnvironmentHelper

  } // end of namespace mars_environment
} // end of namespace bolero

#endif // __MARS_ENVIRONMENT_HELPER_H__
