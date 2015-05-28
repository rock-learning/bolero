/**
 * \file MountainCar.h
 * \author Malte Langosz
 * \brief 
 *
 * Version 0.1
 */

#ifndef __MOUNTAIN_CAR_H__
#define __MOUNTAIN_CAR_H__

#ifdef _PRINT_HEADER_
  #warning "MountainCar.h"
#endif

#include <Environment.h>
#include <cstring> // for memcpy
#include <cassert>
#include <vector>

#define NUM_EPISODES 10

namespace bolero {
  namespace mountain_car {

    class MountainCar : public Environment {

    public:
      MountainCar(lib_manager::LibManager *theManager);
      virtual ~MountainCar();

      CREATE_MODULE_INFO();

      virtual void init();
      virtual void reset();

      /**
       * This functions are used for the controller interfacing a
       * behavior to an environment
       */
      virtual int getNumOutputs() const {return numOutputs;}
      virtual int getNumInputs() const {return numInputs;}

      virtual void getOutputs(double *values, int numOutputs) const;
      virtual void setInputs(const double *values, int numInputs);
      virtual void stepAction();

      virtual bool isEvaluationDone() const;

      virtual int getFeedback(double *feedback) const;

      bool isBehaviorLearningDone() const;

    private:
      int numInputs, numOutputs;
      double *inputs;
      bool evaluationDone;
      int episodeCount;
      double state[2];
      double startStates[NUM_EPISODES][2];
      int stepCount;
      double fitness;

      void StateTransition (double action, double *state);

    }; // end of class definition MountainCar

  } // end of namespace mountain_car
} // end of namespace bolero

#endif // __MOUNTAIN_CAR_H__
