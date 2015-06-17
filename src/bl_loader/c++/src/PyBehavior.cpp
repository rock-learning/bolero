#include <Python.h>
#include "PyBehavior.h"
#include <cassert>
#include <stdexcept>

using namespace std;

namespace bolero { namespace bl_loader {

PyBehavior::PyBehavior(Object& object)
  : object(object)
{}

PyBehavior::PyBehavior(int numInputs, int numOutputs)
  : Behavior(numInputs, numOutputs)
{}

void PyBehavior::setInputs(const double *values, int numInputs) {
  behavior.method("set_inputs").pass(ONEDARRAY).call(&array); // TODO
}

void PyBehavior::getOutputs(double *values, int numOutputs) const {
  behavior.method("get_outputs").pass(ONEDARRAY).call(&array);
}

void PyBehavior::step() {
  behavior.method("step").call();
}

bool PyBehavior::canStep() const
{
  return behavior.method("can_step").call().returnObject().asBool();
}


PyBehavior* PyBehavior::fromPyObject(Object& object) {
  return new PyBehavior(object);;
}

void PyBehavior::setMetaParameters(const MetaParameters &params)
{
  ListBuilder keys = PythonInterpreter::instance().listBuilder();
  ListBuilder values = PythonInterpreter::instance().listBuilder();
  shared_ptr<Object> keysObject, valuesObject;
  for(MetaParameters::const_iterator it = params.begin(); it != params.end();
      ++it)
  {
    keysObject = keys.pass(STRING).call(&it->first);
    valuesObject = values.pass(ONEDARRAY).call(&it->second);
  }

  behavior.method("set_meta_parameters")
    .pass(OBJECT).pass(OBJECT).call(&*keysObject, &*valuesObject);
}

}}
