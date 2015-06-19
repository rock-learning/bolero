#include <Python.h>
#include "PyBehavior.h"
#include <cassert>
#include <stdexcept>


using namespace std;

namespace bolero { namespace bl_loader {

PyBehavior::PyBehavior(shared_ptr<Object> behavior)
  : Behavior(behavior->variable("num_inputs").asInt(),
             behavior->variable("num_outputs").asInt()), behavior(behavior)
{
}

void PyBehavior::setInputs(const double *values, int numInputs) {
  behavior->method("set_inputs").pass(ONEDCARRAY).call(values, numInputs);
}

void PyBehavior::getOutputs(double *values, int numOutputs) const {
  behavior->method("get_outputs").pass(ONEDCARRAY).call(values, numOutputs);
}

void PyBehavior::step() {
  behavior->method("step").call();
}

bool PyBehavior::canStep() const
{
  return behavior->method("can_step").call().returnObject()->asBool();
}


PyBehavior* PyBehavior::fromPyObject(shared_ptr<Object> object) {
  PyBehavior* behavior = new PyBehavior(object);
  return behavior;
}

void PyBehavior::setMetaParameters(const MetaParameters &params)
{
  // We hold copies of all the meta parameters that we use to ensure that the
  // numpy arrays do not refer to memory that has been freed already.
  metaParameters.insert(params.begin(), params.end());

  shared_ptr<ListBuilder> keys = PythonInterpreter::instance().listBuilder();
  shared_ptr<ListBuilder> values = PythonInterpreter::instance().listBuilder();
  for(MetaParameters::const_iterator it = params.begin(); it != params.end();
      ++it)
  {
    keys->pass(STRING).build(&it->first);
    values->pass(ONEDARRAY).build(&metaParameters[it->first]);
  }
  shared_ptr<Object> keysObject = keys->build();
  shared_ptr<Object> valuesObject = values->build();

  behavior->method("set_meta_parameters")
    .pass(OBJECT).pass(OBJECT).call(&*keysObject, &*valuesObject);
}

}}
