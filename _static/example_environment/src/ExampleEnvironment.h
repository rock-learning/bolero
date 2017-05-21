/**
 * \file ExampleEnvironment.h
 * \author Sebastian Klemp
 * \brief An environment containing the Example robot.
 *
 * Version 0.1
 */

#ifndef __Example_ENVIRONMENT_H__
#define __Example_ENVIRONMENT_H__

#ifdef _PRINT_HEADER_
  #warning "ExampleEnvironment.h"
#endif

#include <string>
#include <vector>

#include <MARSEnvironment.h>

namespace bolero {
  namespace Example_environment {

    class ExampleEnvironment: public mars_environment::MARSEnvironment {

    public:
      ExampleEnvironment(lib_manager::LibManager *theManager);

      // LibInterface methods
      int getLibVersion() const {return 1;}
      const std::string getLibName() const {return std::string("Example_environment");}

      virtual void initMARSEnvironment();
      virtual void resetMARSEnvironment();
      virtual void handleMARSError();

      /**
       * This functions are used for the controller interfacing a
       * behavior to an environment
       */
      virtual int getNumInputs() const;
      virtual int getNumOutputs() const;

      virtual void createOutputValues();
      virtual void handleInputValues();

      virtual int getFeedback(double *feedback) const;

      bool isEvaluationDone() const;
      bool isBehaviorLearningDone() const {return false;}

    private:
      const double MAX_TIME;
      const unsigned int numJoints;
      const unsigned int numAllJoints;

      double fitness;
      bool evaluation_done;

      // lists for storing the IDs of the created objects
      std::vector<unsigned long> motorIDs;
      std::vector<unsigned long> sensorIDs;

      void getSensorIDs();
      void getMotorIDs();

    }; // end of class definition ExampleEnvironment
  } // end of namespace Example_environment
} // end of namespace bolero

#endif // __Example_ENVIRONMENT_H__
