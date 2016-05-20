/**
 * \file FunctionApproximation.cpp
 * \author Malte Langosz
 * \brief
 *
 * Version 0.1
 */

#include "FunctionApproximation.h"

#include <configmaps/ConfigData.h>
#include <lib_manager/LibManager.hpp>

#include <cmath>
#include <cstring>

namespace bolero {
  namespace function_approximation {

    FunctionApproximation::FunctionApproximation(lib_manager::LibManager *theManager)
      : Environment(theManager, "function_approximation", 1) {
    }

    FunctionApproximation::~FunctionApproximation() {
      delete[] y;
    }

    void FunctionApproximation::init() {
      configmaps::ConfigMap map;
      configmaps::ConfigMap *map2;
      map = configmaps::ConfigMap::fromYamlFile("learning_config.yml");

      dataFile = "data_to_fit.txt";

      if(map.find("Environment") != map.end()) {
        map2 = &(map["Environment"][0].children);
        if(map2->find("DataFile") != map2->end()) {
          dataFile = (*map2)["DataFile"][0].getString();
        }
        if(map2->find("TestDataFile") != map2->end()) {
          testDataFile = (*map2)["TestDataFile"][0].getString();
        }
      }

      fprintf(stderr, "scan: %s\n", dataFile.c_str());
      readExpData(&fitData, dataFile);
      fprintf(stderr, "scan: %s\n", testDataFile.c_str());
      readExpData(&testData, testDataFile);
      currentData = &fitData;
      y = new double[numInputs];
      reset();
    }

    void FunctionApproximation::reset() {
      error = 0;
      dataPoint = currentData->begin();
    }

    void FunctionApproximation::setTestMode(bool testMode) {
      if(testMode) {
        currentData = &testData;
      }
      else {
        currentData = &fitData;
      }
      dataPoint = currentData->begin();
    }

    void FunctionApproximation::getOutputs(double *values,
                                           int numOutputs) const {
      assert(numOutputs == this->numOutputs);
      for(int i=0; i<numOutputs; ++i) {
        values[i] = dataPoint->inputs[i];
      }
    }

    void FunctionApproximation::setInputs(const double *values,
                                          int numInputs) {
      assert(numInputs == this->numInputs);
      std::memcpy(y, values, sizeof(double)*numInputs);
    }

    void FunctionApproximation::stepAction() {
      // do the evaluation
      for(int i=0; i<numInputs; ++i) {
        error += pow(fabs(dataPoint->outputs[i]-y[i]), 2.0);
      }
      ++dataPoint;
      if(dataPoint == currentData->end()) {
        error /= currentData->size();
        error = sqrt(error);
        //error /= currentData->size()*numOutputs;

        switch(std::fpclassify(error)) {
        case FP_INFINITE:
          error = 10000.0;
          break;
        case FP_NAN:
          error = 10000.0;
          break;
        case FP_NORMAL:
          break;
        case FP_SUBNORMAL:
          break;
        case FP_ZERO:
          error = 0.0;
          break;
        default:
          break;
        }
      }
    }

    int FunctionApproximation::getFeedback(double *feedback) const {
      feedback[0] = error;

      return 1;
    }

    bool FunctionApproximation::isEvaluationDone() const {
      return (dataPoint == currentData->end());
    }

    void FunctionApproximation::readExpData(std::vector<FitData> *expData,
                                            std::string filename) {
      FitData newDataPoint;
      FILE *file = fopen(filename.c_str(), "r");
      char line[1024];
      char *linePtr = line;
      int lineLength;
      double val;
      char text[25];
      int i=0;
      int read;

      if(file) {
        while(readLine(file, line, 1024, &lineLength)) {
          if(i==0) {
            // inputs of testfunction are the outputs of the environmen
            read = sscanf(line, "numInputs: %d\n", &numOutputs);
            assert(read == 1);
          }
          else if(i==1) {
            read = sscanf(line, "numOutputs: %d\n", &numInputs);
            assert(read == 1);
          }
          else {
            newDataPoint.inputs.clear();
            newDataPoint.outputs.clear();
            linePtr = line;
            //fprintf(stderr, "line: %s\n", line);
            for(int i=0; i<numOutputs; ++i) {
              readValue(&linePtr, text, 25, lineLength-(linePtr-line));
              read = sscanf(text, "%lf", &val);
              assert(read == 1);
              //fprintf(stderr, "read: %g\n", val);
              newDataPoint.inputs.push_back(val);
            }
            for(int i=0; i<numInputs; ++i) {
              readValue(&linePtr, text, 25, lineLength-(linePtr-line));
              read = sscanf(text, "%lf", &val);
              assert(read == 1);
              //fprintf(stderr, "read: %g\n", val);
              newDataPoint.outputs.push_back(val);
            }
            expData->push_back(newDataPoint);
          }
          ++i;
        }
      }
      fclose(file);
    }

    void FunctionApproximation::readValue(char **linePtr, char *buf,
                                          int bufSize, int lineLength) {
      int i=0;
      while(i<lineLength && i<bufSize-1 && **linePtr!=';') {
        buf[i++] = **linePtr;
        (*linePtr)++;
      }
      (*linePtr)++;
      buf[i] = '\0';
    }

    bool FunctionApproximation::readLine(FILE *file, char *buf,
                                         int bufSize, int *readLength) {
      char c;
      int i=0;

      c = fgetc(file);
    
      while(c!=EOF && c!='\n' && c!='\r' && i<bufSize-1) {
        buf[i++] = c;
        c = fgetc(file);
      }
      buf[i] = '\0';
      *readLength = i;
      if(i>=bufSize-1)
        fprintf(stderr, "FuncApproxController::readLine - bufferSize to small\n");
      if(c=='\n' || c=='\r') return true;
      return false;
    }


  } // end of namespace function_approximation
} // end of namespace bolero


DESTROY_LIB(bolero::function_approximation::FunctionApproximation);
CREATE_LIB(bolero::function_approximation::FunctionApproximation);
