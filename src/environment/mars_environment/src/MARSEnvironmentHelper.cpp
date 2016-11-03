/**
 * \file MARSEnvironmentHelper.cpp
 * \author Malte Langosz
 * \brief
 *
 * Version 0.1
 */

#include "MARSEnvironmentHelper.h"
#include "MARSThread.h"

#include <configmaps/ConfigData.h>
#include <lib_manager/LibManager.hpp>
#include <mars/interfaces/sim/SimulatorInterface.h>
#include <mars/interfaces/sim/ControlCenter.h>
#include <mars/utils/misc.h>

#include <cmath>
#include <cstring> // needed for memcpy

using namespace configmaps;

namespace bolero {
  namespace mars_environment {

    MARSEnvironmentHelper::MARSEnvironmentHelper(
            lib_manager::LibManager *theManager, const char* name, int version,
            MARSEnvPlugin *marsPlugin)
      : Environment(theManager, name, version), initialized(false),
        marsThread(0), marsPlugin(marsPlugin)
    {
    }

    MARSEnvironmentHelper::~MARSEnvironmentHelper() {
      if(initialized)
        libManager->releaseLibrary("mars_sim");

      //mars::app::MARS::control->sim->StopSimulation();
      marsPlugin->doNotContinue = false;
      marsPlugin->newInputData = true;
      //while(mars::app::MARS::control->sim->isSimRunning()) ;

      if(mars::app::MARS::control)
        mars::app::exit_main(0);

      if(marsThread)
        delete marsThread;
    }

    void MARSEnvironmentHelper::init() {
      int *argc = (int*)malloc(sizeof(int));
      char **argv = (char**)malloc(sizeof(char*)*4);
      *argc = 3;
      argv[0] = (char*)malloc(sizeof(char) * 5);
      argv[1] = (char*)malloc(sizeof(char) * 3);
      argv[2] = (char*)malloc(sizeof(char) * 2);
      strcpy(argv[0], "MARS");
      strcpy(argv[1], "-C");
      strcpy(argv[2], ".");
      argv[3] = NULL;

      ConfigMap map;
      ConfigMap *map2;
      map = ConfigMap::fromYamlFile("learning_config.yml");

      graphicsStepSkip = graphicsUpdateTime = 0;

      bool enableGUI = true;
      if(map.hasKey("Environment")) {
        map2 = map["Environment"];

        if(map2->hasKey("enableGUI"))
          enableGUI = (*map2)["enableGUI"];

        if(map2->hasKey("graphicsUpdateTime"))
          graphicsUpdateTime = (*map2)["graphicsUpdateTime"];
        else
          graphicsUpdateTime = 0u;

        if(map2->hasKey("graphicsStepSkip"))
          graphicsStepSkip = (*map2)["graphicsStepSkip"];
        else
          graphicsStepSkip = 0u;
      }
      if(enableGUI)
        fprintf(stderr, "enableGUI: yes\n");
      else
        fprintf(stderr, "enableGUI: no\n");

      const char *text = getenv("MARS_GRAPHICS_UPDATE_TIME");
      if(text) {
        graphicsUpdateTime = atoi(text);
      }

      marsThread = new MARSThread(libManager, *argc, argv, enableGUI);
      marsThread->setupMARS();

      // load the simulation core_libs
      std::string coreConfigFile = "core_libs.txt";
      if(!enableGUI) {
        coreConfigFile = "core_libs-nogui.txt";
      }

      FILE *testFile = fopen(coreConfigFile.c_str(), "r");
      if(testFile) {
        fclose(testFile);
        libManager->loadConfigFile(coreConfigFile.c_str());
      } else {
        fprintf(stderr, "Loading default core libraries...\n");
        libManager->loadLibrary("cfg_manager");
        libManager->loadLibrary("data_broker");
        libManager->loadLibrary("mars_sim");
        libManager->loadLibrary("mars_scene_loader");
        libManager->loadLibrary("mars_entity_factory");
        libManager->loadLibrary("mars_smurf");
        libManager->loadLibrary("mars_smurf_loader");
        if(enableGUI) {
          libManager->loadLibrary("main_gui");
          libManager->loadLibrary("mars_graphics");
          libManager->loadLibrary("mars_gui");
        }
      }

      fprintf(stderr, "Loading default additional libraries...\n");
      // loading errors will be silent for the following optional libraries
      if(enableGUI) {
        libManager->loadLibrary("connexion_plugin", NULL, true);
        libManager->loadLibrary("data_broker_gui", NULL, true);
        libManager->loadLibrary("cfg_manager_gui", NULL, true);
        libManager->loadLibrary("lib_manager_gui", NULL, true);
      }

      // load the simulation other_libs:
      std::string otherConfigFile = "other_libs.txt";
      testFile = fopen(otherConfigFile.c_str() , "r");
      if(testFile) {
        fclose(testFile);
        libManager->loadConfigFile(otherConfigFile);
      }

      marsThread->startMARS();

      mars::interfaces::SimulatorInterface *mars;
      mars = libManager->getLibraryAs<mars::interfaces::SimulatorInterface>("mars_sim");
      assert(mars);
      marsPlugin->control = mars->getControlCenter();
      mars::interfaces::pluginStruct newplugin;
      newplugin.name = getLibName();
      newplugin.p_interface = dynamic_cast<mars::interfaces::PluginInterface*>(marsPlugin);
      newplugin.p_destroy = 0;
      mars->addPlugin(newplugin);

      // this will call the init function in the mars plugin
      mars::app::MARS::control->sim->finishedDraw();

      // after the init call of the mars plugin we know the number of
      // inputs and outputs
      marsPlugin->inputs = new double[getNumInputs()];
      marsPlugin->outputs = new double[getNumOutputs()];

      initialized = true;
    }

    void MARSEnvironmentHelper::reset() {
      marsPlugin->waitForReset = true;
      marsPlugin->doNotContinue = false;
      marsPlugin->newOutputData = false;

      for(int i=0; i<getNumOutputs(); ++i) {
        marsPlugin->outputs[i] = 0.0;
      }

      mars::app::MARS::control->sim->resetSim(false);
      mars::app::MARS::control->sim->finishedDraw();
    }

    void MARSEnvironmentHelper::getOutputs(double *values,
                                     int numOutputs) const {
      assert(numOutputs == getNumOutputs());

      marsPlugin->doNotContinue = false;

      /*
      while(!marsPlugin->newOutputData) {
        if(marsPlugin->waitForReset) {
          if(marsThread->myApp) {
            marsThread->myApp->processEvents();
          }
          else {
            mars::app::MARS::control->sim->finishedDraw();
          }
          //if(!marsPlugin->waitForReset) mars::app::MARS::control->sim->StartSimulation();
        }
      }
*/

      marsPlugin->dataMutex.lock();
      memcpy(values, marsPlugin->outputs,
             sizeof(double)*numOutputs);
      marsPlugin->dataMutex.unlock();
      marsPlugin->newOutputData = false;
      marsPlugin->finishedStep = false;

      marsPlugin->doNotContinue = true;
    }

    void MARSEnvironmentHelper::setInputs(const double *values,
                                    int numInputs) {
      assert(numInputs == getNumInputs());

      marsPlugin->dataMutex.lock();
      memcpy(marsPlugin->inputs, values, sizeof(double)*numInputs);
      marsPlugin->dataMutex.unlock();
      marsPlugin->doNotContinue = true;
      marsPlugin->newInputData = true;
    }

    void MARSEnvironmentHelper::stepAction() {
      static int updateCount = graphicsStepSkip;
      // do the evaluation

      //mars::app::MARS::control->sim->step();

      /*
      while(!marsPlugin->finishedStep) { }
      */

      while(!marsPlugin->newOutputData) {
        mars::app::MARS::control->sim->step(true);
      }

      if(--updateCount < 0) {
	updateCount = graphicsStepSkip;
	if(marsThread->myApp) {
	  marsThread->myApp->processEvents();
	}
	else {
	  mars::app::MARS::control->sim->finishedDraw();
	}
	if(graphicsUpdateTime) {
	  mars::utils::msleep(graphicsUpdateTime);
	}
      }
    }

  } // end of namespace mars_environment
} // end of namespace bolero
