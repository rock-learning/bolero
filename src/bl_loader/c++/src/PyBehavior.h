#ifndef PY_BEHAVIOR_H
#define PY_BEHAVIOR_H

#include <PythonInterpreter.hpp>
#include <string>
#include <Behavior.h>
#ifdef NO_TR1
#include <unordered_map>
#else
#include <tr1/unordered_map>
#endif
#include <vector>

namespace bolero { namespace bl_loader {

class PyBehavior : public bolero::Behavior {
public:
  // this is the prefered way of construction
  static PyBehavior* fromPyObject(shared_ptr<Object> object);

  void setInputs(const double *values, int numInputs);
  void getOutputs(double *values, int numOutputs) const;

  /**
   * Meta-parameters could be the goal, obstacles, etc.
   * Each parameter is a list of doubles identified by a key.
   */
#ifdef NO_TR1
  typedef std::unordered_map<std::string, std::vector<double> > MetaParameters;
#else
  typedef std::tr1::unordered_map<std::string, std::vector<double> > MetaParameters;
#endif
  void setMetaParameters(const MetaParameters &params);

  void step();
  bool canStep() const;

private:
  PyBehavior(shared_ptr<Object> behavior);
  // disallow copying and assigning
  PyBehavior(const PyBehavior&);
  PyBehavior& operator=(const PyBehavior&);
  shared_ptr<Object> behavior;
}; /* end of class PyBehavior */

}}

#endif /* PY_BEHAVIOR_H */
