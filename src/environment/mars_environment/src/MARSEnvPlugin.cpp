/**
 * \file MARSEnvPlugin.h
 * \author Malte Langosz
 * \brief see header
 *
 * Version 0.1
 */


#include "MARSEnvPlugin.h"
#include <mars/data_broker/DataBrokerInterface.h>
#include <mars/data_broker/DataPackage.h>
#include <mars/interfaces/sim/SimulatorInterface.h>
#include <mars/interfaces/sim/ControlCenter.h>
#include <mars/cfg_manager/CFGManagerInterface.h>
#include <mars/utils/mathUtils.h>
#include <mars/utils/misc.h>
#include <math.h>
#include <cassert>


// test time = 10sec = 10000ms -> 10000/20 ticks -> 500 ticks
#define MAX_TIME 10000

using namespace mars;
using namespace mars::interfaces;
using namespace mars::utils;
using namespace configmaps;

namespace bolero {  
  namespace mars_environment {  

    MARSEnvPlugin::MARSEnvPlugin()
        : mars::interfaces::PluginInterface(NULL), r(0) {
      waitForReset = false;
      leftTime = 0.0;
      nextUpdate = 0.0;
      stepTimeMs = 20.0;

      finishedStep = false;
      newInputData = false;
      newOutputData = false;
      doNotContinue = true;
    }
  
    void MARSEnvPlugin::init() {
      assert(control);

      initMARSEnvironment();

      r = new MARSReceiver(this);
      control->dataBroker->registerTriggeredReceiver(
        r, "mars_sim", "simTime", "mars_sim/postPhysicsUpdate");
      control->dataBroker->registerTriggeredReceiver(
        r, "mars_sim", "simTime", "mars_sim/prePhysicsUpdate");
    }

    void MARSEnvPlugin::reset() {
      leftTime = 0;
      nextUpdate = -1;
      resetMARSEnvironment();
      waitForReset = false;
    }

    MARSEnvPlugin::~MARSEnvPlugin() {
      if(r) delete r;
    }

    void MARSEnvPlugin::update(sReal time_ms) {
      leftTime += time_ms;
      this->time_ms = time_ms;
      if(time_ms > stepTimeMs) {
        LOG_WARN("MARSEnvPlugin: The simulation step time is greater than the "
                 "desired update time of the behavior.");
      }
      if(leftTime >= nextUpdate) {
        nextUpdate += stepTimeMs;
        update();
      }
    }

    void MARSEnvPlugin::update() {
      if(!waitForReset) {
        dataMutex.lock();
        createOutputValues();
        dataMutex.unlock();
        newOutputData=true;

        dataMutex.lock();
        newInputData = false;
        handleInputValues();
        dataMutex.unlock();
      }
    }

    void MARSEnvPlugin::handleError() {
      handleMARSError();
    }

    void MARSEnvPlugin::receive() {}

  } // end of namespace mars_environment
} // end of namespace bolero
