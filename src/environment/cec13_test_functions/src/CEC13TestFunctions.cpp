/**
 * \file CEC13TestFunctions.cpp
 * \author Malte Langosz
 * \brief
 *
 * Version 0.1
 */

#include "CEC13TestFunctions.h"

#include <configmaps/ConfigData.h>
#include <lib_manager/LibManager.hpp>

#include <cmath>
#include <cstring>

void test_func(double *, double *,int,int,int);

double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag;

using namespace configmaps;

namespace bolero {
  namespace cec13_test_functions {

    CEC13TestFunctions::CEC13TestFunctions(lib_manager::LibManager *theManager)
      : Environment(theManager, "cec13_test_functions", 1) {
      functionValue = 0.;
    }

    CEC13TestFunctions::~CEC13TestFunctions() {
      delete[] x;

      free(y);
      free(z);
      free(M);
      free(OShift);
      free(x_bound);
    }

    void CEC13TestFunctions::init(std::string config) {
      ConfigMap map;
      ConfigMap *map2;

      dimension = 10;
      testFunction = 1;

      if(config != "") {
        map = ConfigMap::fromYamlString(config);
        if(map.find("Environment") != map.end()) {
          map2 = map["Environment"];
        } else {
          map2 = &map;
        }
        dimension = map2->get("Dimension", dimension);
        assert(dimension>0);
        testFunction = map2->get("CEC13TestFunction", testFunction);
        assert(testFunction > 0 && testFunction < 29);
      }

      x = new double[dimension];
    }

    void CEC13TestFunctions::reset() {
    }


    int CEC13TestFunctions::getNumInputs() const {
      return dimension;
    }

    void CEC13TestFunctions::setInputs(const double *values,
                                      int numInputs) {
      assert(numInputs == dimension);

      std::memcpy(x, values, sizeof(double)*dimension);
      // we always get values between 0 and 1 from the optimizer, so we
      // scale them to an expected range
      for(int i=0; i<dimension; ++i) {
        x[i] -= 0.5;
        x[i] *= 200;
      }
    }

    void CEC13TestFunctions::stepAction() {
      test_func(x, &functionValue, dimension, 1, testFunction);
    }

    int CEC13TestFunctions::getFeedback(double *feedback) const {
      *feedback = functionValue;
      return 1;
    }

    bool CEC13TestFunctions::isEvaluationDone() const {
      if(testFunction > 0) return true;
      return false;
    }

    bool CEC13TestFunctions::isBehaviorLearningDone() const {
      if(fabs(functionValue + 1400 - 100*(testFunction-1)) < 0.0001) {
        return true;
      }
      return false;
    }

  } // end of namespace cec13_test_functions
} // end of namespace bolero


DESTROY_LIB(bolero::cec13_test_functions::CEC13TestFunctions);
CREATE_LIB(bolero::cec13_test_functions::CEC13TestFunctions);
