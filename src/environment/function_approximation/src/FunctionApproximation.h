/**
 * \file FunctionApproximation.h
 * \author Malte Langosz
 * \brief 
 *
 * Version 0.1
 */

#ifndef __FUNCTION_APPROXIMATION_H__
#define __FUNCTION_APPROXIMATION_H__

#ifdef _PRINT_HEADER_
  #warning "FunctionApproximation.h"
#endif

#include <Environment.h>
#include <cstring> // for memcpy
#include <cassert>
#include <vector>

namespace bolero {
  namespace function_approximation {

    class FunctionApproximation : public Environment {

      struct FitData {
        std::vector<double> inputs;
        std::vector<double> outputs;
      };

    public:
      FunctionApproximation(lib_manager::LibManager *theManager);
      virtual ~FunctionApproximation();

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

      void setTestMode(bool testMode);

      virtual int getFeedback(double *feedback) const;

      bool isBehaviorLearningDone() const {return false;}

    private:
      std::vector<FitData>::iterator dataPoint;
      std::vector<FitData> *currentData;
      std::vector<FitData> fitData, fitData2;
      std::vector<FitData> testData;
      int numInputs, numOutputs;
      double error;
      std::string dataFile, dataFile2;
      std::string testDataFile;
      double *y;
      unsigned long numEvaluationsToSwitch, evaluationCount, evalCount, evaluateRunX;

      void readExpData(std::vector<FitData> *expData, std::string filename);
      void readValue(char **linePtr, char *buf, int bufSize, int lineLength);
      bool readLine(FILE *file, char *buf, int bufSize, int *readLength);

    }; // end of class definition FunctionApproximation

  } // end of namespace function_approximation
} // end of namespace bolero

#endif // __FUNCTION_APPROXIMATION_H__
