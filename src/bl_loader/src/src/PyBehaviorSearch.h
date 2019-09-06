#ifndef PY_BEHAVIORSEARCH_H
#define PY_BEHAVIORSEARCH_H

#include <PythonInterpreter.hpp>
#include <string>
#include <BehaviorSearch.h>
#include "PyLoadable.h"

namespace bolero { namespace bl_loader {

class PyBehaviorSearch : public BehaviorSearch, public PyLoadable {
public:
  PyBehaviorSearch(lib_manager::LibManager *theManager,
                   const std::string libName, int libVersion);
  ~PyBehaviorSearch();

  void init(int numInputs, int numOutputs, std::string config="");
  bolero::Behavior* getNextBehavior();
  bolero::Behavior* getBestBehavior();
  void setEvaluationFeedback(const double *feedbacks,
                             int numFeedbacks);
  void writeResults(const std::string &resultPath);
  bolero::Behavior* getBehaviorFromResults(const std::string &resultPath);
  bool isBehaviorLearningDone() const;

  bool implementsBatchComputing() const {return true;}
  std::string getBehaviorBatch() const;
  void setBatchFeedback(const double* batchFeedback, int numFeedbacksPerBatch, int batchSize);
  Behavior* getBehaviorFromString(std::string &behaviorString);

private:
  std::string className;
  shared_ptr<Object> behaviorSearch;
  Behavior* behavior;
  Behavior* bestBehavior;
}; /* end of class PyBehaviorSearch */

}}
#endif /* PY_BEHAVIORSEARCH_H */
