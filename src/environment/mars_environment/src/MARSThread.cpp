/**
 * \file MARSThread.h
 * \author Malte Langosz
 * \brief 
 *
 * Version 0.1
 */

#include "MARSThread.h"
#include <stdio.h>

namespace bolero {
  namespace mars_environment {

    MARSThread::MARSThread(lib_manager::LibManager *theManager,
                           int &argc, char** argv,
                           bool enableGUI) : libManager(theManager),
                                             argc(argc), argv(argv),
                                             simulation(0),
                                             simulationStarted(false),
                                             enableGUI(enableGUI) {
      if(enableGUI) {
        myApp = new mars::app::MyApp(argc, argv);
      } else {
        myApp = NULL;
      }
    }

    MARSThread::~MARSThread() {
      if(simulation)
        delete simulation;
    }

    void MARSThread::run() {
      int state = simulation->runWoQApp();

      delete simulation;
      simulation = 0;
      exit(state);
    }

    void MARSThread::setupMARS() {
      simulation = new mars::app::MARS(libManager);
      if(enableGUI) {
        simulation->needQApp = true;
      } else {
        simulation->needQApp = false;
      }
      simulation->readArguments(argc, argv);
      simulation->init();
    }

    void MARSThread::startMARS() {
      simulation->start(argc, argv, false, false);
      simulationStarted = true;
    }

  } // end of namespace mars_environment
} // end of namespace bolero
