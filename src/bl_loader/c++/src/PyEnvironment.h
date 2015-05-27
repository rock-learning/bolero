#ifndef PyEnvironment_H
#define PyEnvironment_H

#include <string>
#include <Environment.h>
#include "Helper.h"
#include "PyLoadable.h"

namespace bolero { namespace bl_loader {

class PyEnvironment : public Environment, public PyLoadable{
public:
  PyEnvironment(lib_manager::LibManager *theManager,
                 const std::string libName, int libVersion);
  virtual ~PyEnvironment();

  void init();
  void reset();

  int getNumInputs() const;
  int getNumOutputs() const;

  void getOutputs(double *values, int numOutputs) const;
  void setInputs(const double *values, int numInputs);
  void stepAction();

  bool isEvaluationDone() const;

  // returns if a reward was assigned to the pointer parameter
  // for the whole evaluation
  int getFeedback(double *feedback) const;


  bool isBehaviorLearningDone() const;

private:
  PyObjectPtr py_environment;
  py_callable_info_t get_outputs;
  py_callable_info_t set_inputs;
  py_callable_info_t get_step_feedback;
  py_callable_info_t get_feedbacks;

}; /* end of class PyEnvironment */

}}
#endif /* PyEnvironment_H */
