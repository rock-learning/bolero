/**
 * \file MARSEnvPlugin.h
 * \author Malte Langosz
 * \brief 
 *
 * Version 0.1
 */

#ifndef __MARS_ENV_PLUGIN_H__
#define __MARS_ENV_PLUGIN_H__

#ifdef _PRINT_HEADER_
  #warning "MARSEnvPlugin.h"
#endif

#include "MARSReceiver.h"

#include <mars/interfaces/sim/PluginInterface.h>
#include <mars/utils/Mutex.h>
#include <mars/cfg_manager/CFGManagerInterface.h>

namespace bolero {
  namespace mars_environment {

    class MARSEnvPlugin: public mars::interfaces::PluginInterface {
      /// @private
      friend class MARSEnvironmentHelper;
    public:
      MARSEnvPlugin();
      ~MARSEnvPlugin();

      void init();

      void reset();
      void update(mars::interfaces::sReal time_ms);
      void handleError();

      void receive();

      virtual void initMARSEnvironment() = 0;
      virtual void resetMARSEnvironment() = 0;
      virtual void createOutputValues() = 0;
      virtual void handleInputValues() = 0;
      virtual bool isEvaluationDone() const = 0;
      virtual void handleMARSError() = 0;

    protected:
      std::string configString;
      MARSReceiver *r;
      mutable bool doNotContinue;
      mutable bool newOutputData;
      bool newInputData;

      mars::interfaces::sReal time_ms;
      double *inputs;
      double *outputs;
      double contact;
      mutable bool waitForReset;
      mutable bool finishedStep;

      double leftTime;
      double nextUpdate;
      double stepTimeMs;
      mutable mars::utils::Mutex dataMutex;

      void update();
    }; // end of class definition MARSEnvPlugin

  } // end of namespace mars_environment
} // end of namespace bolero

#endif // __MARS_ENV_PLUGIN_H__
