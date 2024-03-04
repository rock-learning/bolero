#ifndef PyEnvironment_H
#define PyEnvironment_H

#include <PythonInterpreter.hpp>
#include <string>
#include <Environment.h>
#include <PyLoadable.h>

namespace bolero { namespace bl_loader {

class PyEnvironment : public Environment, public PyLoadable {
public:
  PyEnvironment(lib_manager::LibManager *theManager, const std::string libName,
                int libVersion);

  void init(std::string config="");
  void reset();

  int getNumInputs() const;
  int getNumOutputs() const;

  void getOutputs(double *values, int numOutputs) const;
  void setInputs(const double *values, int numInputs);
  void stepAction();

  bool isEvaluationDone() const;
  bool isEvaluationAborted() const;

  /** Get feedbacks.
   * @param[out] feedback array, will be filled with feedback values
   * @return Number of rewards that was assigned to the pointer parameter
   * for the whole evaluation
   */
  int getFeedback(double *feedback) const;

  bool isBehaviorLearningDone() const;

private:
  shared_ptr<Object> environment;
}; /* end of class PyEnvironment */

}}
#endif /* PyEnvironment_H */
