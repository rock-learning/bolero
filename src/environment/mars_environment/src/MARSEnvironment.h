/**
 * \file MARSEnvironment.h
 * \author Malte Langosz
 * \brief 
 *
 * Version 0.1
 */

#ifndef __MARS_ENVIRONMENT_H__
#define __MARS_ENVIRONMENT_H__

#ifdef _PRINT_HEADER_
  #warning "MARSEnvironment.h"
#endif

#include "MARSEnvironmentHelper.h"
#include "MARSEnvPlugin.h"


namespace bolero {
  namespace mars_environment {

    class MARSEnvironment : public MARSEnvironmentHelper, public MARSEnvPlugin {

    public:
      MARSEnvironment(lib_manager::LibManager *theManager,
                      const char* name, int version);
      virtual ~MARSEnvironment();

    }; // end of class definition MARSEnvironment

  } // end of namespace mars_environment
} // end of namespace bolero

#endif // __MARS_ENVIRONMENT_H__
