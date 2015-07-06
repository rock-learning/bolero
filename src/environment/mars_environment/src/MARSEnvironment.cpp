/**
 * \file MARSEnvironment.cpp
 * \author Malte Langosz
 * \brief 
 *
 * Version 0.1
 */

#include "MARSEnvironment.h"

namespace bolero {
  namespace mars_environment {

    MARSEnvironment::MARSEnvironment(lib_manager::LibManager *theManager,
                                     const char* name, int version)
      : Environment(theManager, name, version),
        MARSEnvironmentHelper(theManager, name, version, this) {
    }

    MARSEnvironment::~MARSEnvironment() {
    }

  } // end of namespace mars_environment
} // end of namespace bolero
