/**
 * \file CEC13TestFunctions.h
 * \author Malte Langosz
 * \brief This simple test problem searches for a minimum value in a test function.
 *
 * Version 0.1
 */

#ifndef __CEC13_TEST_FUNCTIONS_H__
#define __CEC13_TEST_FUNCTIONS_H__

#ifdef _PRINT_HEADER_
  #warning "CEC13TestFunctions.h"
#endif

#include <Environment.h>
#include <cstring> // for memcpy
#include <cassert>
#include <cstring>

namespace bolero {
  namespace cec13_test_functions {

    class CEC13TestFunctions : public Environment {

    public:
      CEC13TestFunctions(lib_manager::LibManager *theManager);
      virtual ~CEC13TestFunctions();

      CREATE_MODULE_INFO();

      virtual void init();
      virtual void reset();

      /**
       * This functions are used for the controller interfacing a
       * behavior to an environment
       */
      virtual int getNumOutputs() const {
        if(testFunction==0) return dimension;
        return 0;
      }
      virtual int getNumInputs() const;
      virtual void getOutputs(double *values, int numOutputs) const {
        if(testFunction==0) {
          assert(numOutputs == dimension);
          memcpy(values, in, sizeof(double)*dimension);
        }
      }
      virtual void setInputs(const double *values, int numInputs);
      virtual void stepAction();

      virtual bool isEvaluationDone() const;

      // returns if a reward was assigned to the pointer parameter
      // for the whole evaluation
      virtual int getFeedback(double *feedback) const;

      bool isBehaviorLearningDone() const {return false;}

    private:
      int dimension, testFunction;
      double *in;
      double *x;
      double functionValue;

    }; // end of class definition CEC13TestFunctions

  } // end of namespace cec13_test_functions
} // end of namespace bolero

#endif // __CEC13_TEST_FUNCTIONS_H__
