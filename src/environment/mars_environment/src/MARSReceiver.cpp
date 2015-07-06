/**
 * \file MARSReceiver.h
 * \author Malte Langosz
 * \brief see header
 *
 * Version 0.1
 */


#include "MARSReceiver.h"
#include "MARSEnvPlugin.h"

namespace bolero {  
  namespace mars_environment {  

    MARSReceiver::MARSReceiver(MARSEnvPlugin *p) : p(p) {
    }
  
    MARSReceiver::~MARSReceiver() {
    }

    void MARSReceiver::receiveData(const mars::data_broker::DataInfo &info,
                                    const mars::data_broker::DataPackage &package, int da) {
      p->receive();
    }

  } // end of namespace mars_environment
} // end of namespace bolero
