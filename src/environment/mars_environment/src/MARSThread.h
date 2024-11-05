/**
 * \file MARSThread.h
 * \author Malte Langosz
 * \brief 
 *
 * Version 0.1
 */

#ifndef __MARS_THREAD_H__
#define __MARS_THREAD_H__

#ifdef _PRINT_HEADER_
  #warning "MARSThread.h"
#endif

#include <mars/app/MARS.h>
#include <mars/app/MyApp.h>
#include <mars/utils/misc.h>
#include <mars/utils/Thread.h>


namespace bolero {
  namespace mars_environment {

    class MARSThread : public mars::utils::Thread {

    public:
      MARSThread(lib_manager::LibManager *theManager,
                 int &argc, char** argv, bool enableGUI);
      ~MARSThread();

      lib_manager::LibManager *libManager;
      int argc;
      char **argv;
      mars::app::MARS *simulation;
      mars::app::MyApp *myApp;
      bool simulationStarted;
      bool enableGUI;

      void run();
      void setupMARS();
      void startMARS();
    };

  } // end of namespace mars_environment
} // end of namespace bolero

#endif // __MARS_THREAD_H__
