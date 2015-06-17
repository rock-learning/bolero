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

  void init(int numInputs, int numOutputs);
  bolero::Behavior* getNextBehavior();
  void setEvaluationFeedback(const double *feedbacks,
                             int numFeedbacks);
  void writeResults(const std::string &resultPath);
  bolero::Behavior* getBehaviorFromResults(const std::string &resultPath);
  bool isBehaviorLearningDone() const;

private:
  std::string className;
  PyObjectPtr behaviorSearch;

}; /* end of class PyBehaviorSearch */

}}
#endif /* PY_BEHAVIORSEARCH_H */
