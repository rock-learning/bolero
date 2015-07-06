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


// test time = 10sec = 10000ms -> 10000/20 ticks -> 500 ticks
#define MAX_TIME 10000

using namespace mars;
using namespace mars::interfaces;
using namespace mars::utils;
using namespace configmaps;

namespace bolero {  
  namespace mars_environment {  

    MARSEnvPlugin::MARSEnvPlugin( ) : mars::interfaces::PluginInterface(NULL) {
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

      ConfigMap map;
      ConfigMap *map2;
      map = ConfigMap::fromYamlFile("learning_config.yml");

      if(map.find("Environment Parameters") != map.end()) {
        map2 = &(map["Environment Parameters"][0].children);

        if(map2->find("calc_ms") != map2->end()) {
          double dValue = (*map2)["calc_ms"][0].getDouble();
          if(control->cfg) {
            control->cfg->setPropertyValue("Simulator", "calc_ms", "value",
                                           dValue);
          }
        }

        if(map2->find("stepTimeMs") != map2->end()) {
          stepTimeMs = (*map2)["stepTimeMs"][0].getDouble();
        }
      }

      initMARSEnvironment();

      r = new MARSReceiver(this);
      control->dataBroker->registerTriggeredReceiver(r, "mars_sim",
                                                     "simTime",
                                                     "mars_sim/postPhysicsUpdate");
      control->dataBroker->registerTriggeredReceiver(r, "mars_sim",
                                                     "simTime",
                                                     "mars_sim/prePhysicsUpdate");
    }

    void MARSEnvPlugin::reset() {
      leftTime = 0;
      nextUpdate = -1;
      resetMARSEnvironment();
      waitForReset = false;
    }

    MARSEnvPlugin::~MARSEnvPlugin() {
      delete r;
    }


    void MARSEnvPlugin::update(sReal time_ms) {
      //if(waitForReset) return;
      leftTime += time_ms;
      this->time_ms = time_ms;
      if(time_ms > stepTimeMs) {
        LOG_WARN("MARSEnvPlugin: The simulation step time is greater than the desired update time of the behavior.");
      }
      if(leftTime >= nextUpdate) {
        nextUpdate += stepTimeMs;
        update();
      }
    }

    void MARSEnvPlugin::update() {
      if(1 || !waitForReset) {
        dataMutex.lock();
        createOutputValues();
        dataMutex.unlock();
        
        newOutputData=true;

        //while(!newInputData) { }

        //if(!waitForReset) {
          dataMutex.lock();
          newInputData = false;
          handleInputValues();
          //evaluate = !isEvaluationDone();
          dataMutex.unlock();
          //}
      }
      else {
        /* do we need this? */
        /*
        newInputData=true;
        while(!newOutputData) { }
        newOutputData = false;
        */
      }
    }

    void MARSEnvPlugin::handleError() {
      handleMARSError();
    }

    void MARSEnvPlugin::receive() {
      return;
      finishedStep = true;
      while(doNotContinue && !waitForReset && control->sim->isSimRunning()) { mars::utils::msleep(1); }
    }

  } // end of namespace mars_environment
} // end of namespace bolero
