/**
 * \file MountainCar.cpp
 * \author Malte Langosz
 * \brief
 *
 * Version 0.1
 */

#include "MountainCar.h"

#include <configmaps/ConfigData.h>
#include <lib_manager/LibManager.hpp>

#include <cmath>
#include <cstring>
#include <cstdlib>

// state0 x wagen
// state1 v wagen


#define MAXSTEPS  			2500	 // maximum number of steps before terminal state

namespace bolero {
  namespace mountain_car {

    MountainCar::MountainCar(lib_manager::LibManager *theManager)
      : Environment(theManager, "mountain_car", 1), inputs(NULL) {

      srand(123654);

      for(int i=0; i<NUM_EPISODES; ++i) {
        startStates[i][0] = ((((double)rand())/RAND_MAX)*1.6)-1.12;
        startStates[i][1] = ((((double)rand())/RAND_MAX)*0.14)-0.07;
      }
    }

    MountainCar::~MountainCar() {
      if(inputs) {
        delete[] inputs;
      }
    }

    void MountainCar::init(std::string config) {
      numOutputs = 2;
      numInputs = 3;
      inputs = new double[numInputs];

      reset();
    }

    void MountainCar::reset() {
      if(NUM_EPISODES > 1) {

        for(int i=0; i<NUM_EPISODES; ++i) {
          startStates[i][0] = ((((double)rand())/RAND_MAX)*1.6)-1.12;
          startStates[i][1] = ((((double)rand())/RAND_MAX)*0.14)-0.07;
        }

        state[0] = startStates[0][0];
        state[1] = startStates[0][1];
      }
      else {
        state[0] = ((((double)rand())/RAND_MAX)*1.6)-1.12;
        state[1] = ((((double)rand())/RAND_MAX)*0.14)-0.07;
      }
      stepCount = 0;
      episodeCount = 0;
      fitness = 0.0;
      evaluationDone = false;
    }


    void MountainCar::getOutputs(double *values,
                                 int numOutputs) const {
      assert(numOutputs == this->numOutputs);
      for(int i=0; i<numOutputs; ++i) {
        values[i] = state[i];
      }
    }

    void MountainCar::setInputs(const double *values,
                                int numInputs) {
      assert(numInputs == this->numInputs);
      std::memcpy(inputs, values, sizeof(double)*numInputs);
    }

    void MountainCar::stepAction() {
      double action = 0.0;
      if(inputs[0] > inputs[1]) {
        if(inputs[0] > inputs[2]) {
          action = 1.0;
        }
        else {
          action = -1.0;
        }
      }
      else if(inputs[1] < inputs[2]) {
        action = -1.0;
      }
      StateTransition (action, state);
      stepCount += 1;
      if(state[0] >= 0.49 || stepCount >= MAXSTEPS) {
        if(++episodeCount < NUM_EPISODES) {
          fitness += stepCount/(double)NUM_EPISODES;
          //fprintf(stderr, "\t%d %g", stepCount, fitness);
          stepCount = 0;
          state[0] = startStates[episodeCount][0];
          state[1] = startStates[episodeCount][1];
        }
        else {
          fitness += stepCount/(double)NUM_EPISODES;
          evaluationDone = true;
        }
      }
    }

    int MountainCar::getFeedback(double *feedback) const {
      feedback[0] = fitness;
      return 1;
    }

    bool MountainCar::isEvaluationDone() const {
      bool finished = false;
      if(evaluationDone) {
        finished = true;
      }
      //if(finished) fprintf(stderr, "finished ind.. %d\n", stepCount);
      return finished;
    }

    bool MountainCar::isBehaviorLearningDone() const {
      return false;
    }

    ////////////////////////////////////////////////////////////////////////
    // State transititon function
    ////////////////////////////////////////////////////////////////////////

    void MountainCar::StateTransition (double action, double *state) {
      double dydx[2];

      if(action > 0.1) action = 1.0;
      else if(action < -0.1) action = -1.0;
      else action = 0.0;

      dydx[1] = state[1]+0.001*action-0.0025*cos(3*state[0]);
      if(dydx[1] > 0.07) dydx[1] = 0.07;
      else if(dydx[1] < -0.07) dydx[1] = -0.07;

      dydx[0] = state[0] + dydx[1];
      if(dydx[0] > 0.5) dydx[0] = 0.5;
      else if(dydx[0] < -1.2) {
        dydx[0] = -1.2;
        dydx[1] = 0.0;
      }

      state[0] = dydx[0];
      state[1] = dydx[1];
    }

  } // end of namespace mountain_car
} // end of namespace bolero


DESTROY_LIB(bolero::mountain_car::MountainCar);
CREATE_LIB(bolero::mountain_car::MountainCar);
