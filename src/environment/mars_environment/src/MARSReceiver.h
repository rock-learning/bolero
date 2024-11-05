/**
 * \file MARSReceiver.h
 * \author Malte Langosz
 * \brief 
 *
 * Version 0.1
 */

#ifndef __MARS_RECEIVER_H__
#define __MARS_RECEIVER_H__

#ifdef _PRINT_HEADER_
  #warning "MARSReceiver.h"
#endif

#include <mars/data_broker/ReceiverInterface.h>

namespace bolero {
  namespace mars_environment {

    class MARSEnvPlugin;

    class MARSReceiver: public mars::data_broker::ReceiverInterface {

    public:
      MARSReceiver(MARSEnvPlugin *p);
      ~MARSReceiver();

      virtual void receiveData(const mars::data_broker::DataInfo &info,
                               const mars::data_broker::DataPackage &package,
                               int da);

    protected:
      MARSEnvPlugin *p;

    }; // end of class definition MARSReceiver

  } // end of namespace mars_environment
} // end of namespace bolero

#endif // __MARS_RECEIVER_H__
