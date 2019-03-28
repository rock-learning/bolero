#ifndef PY_BEHAVIOR_H
#define PY_BEHAVIOR_H

#include <PythonInterpreter.hpp>
#include <string>
#include <Behavior.h>
#include <vector>


namespace bolero { namespace bl_loader {

class PyBehavior : public bolero::Behavior {
public:
  // this is the prefered way of construction
  static PyBehavior* fromPyObject(shared_ptr<Object> object);
  void init(int numInputs, int numOutputs);

  void setInputs(const double *values, int numInputs);
  void getOutputs(double *values, int numOutputs) const;
  void setMetaParameters(const MetaParameters& params);

  void step();
  bool canStep() const;

private:
  PyBehavior(shared_ptr<Object> behavior);
  // disallow copying and assigning
  PyBehavior(const PyBehavior&);
  PyBehavior& operator=(const PyBehavior&);
  shared_ptr<Object> behavior;
  MetaParameters metaParameters;
}; /* end of class PyBehavior */

}}

#endif /* PY_BEHAVIOR_H */
