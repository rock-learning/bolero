#include <Python.h>
#include "PyBehavior.h"
#include <cassert>
#include <stdexcept>


using namespace std;

namespace bolero { namespace bl_loader {

PyBehavior::PyBehavior(shared_ptr<Object> behavior)
  : Behavior(), behavior(behavior)
{
}

void PyBehavior::init(int numInputs, int numOutputs)
{
  pthread_mutex_lock (&PythonInterpreter::mutex);
  Behavior::init(numInputs, numOutputs);
  behavior->method("init").pass(INT).pass(INT).call(numInputs, numOutputs);
  pthread_mutex_unlock (&PythonInterpreter::mutex);
}

void PyBehavior::setInputs(const double *values, int numInputs) {
  pthread_mutex_lock (&PythonInterpreter::mutex);
  behavior->method("set_inputs").pass(ONEDCARRAY).call(values, numInputs);
  pthread_mutex_unlock (&PythonInterpreter::mutex);
}

void PyBehavior::setTargetState(const double *values, int numInputs) {
  pthread_mutex_lock (&PythonInterpreter::mutex);
  behavior->method("set_target_state").pass(ONEDCARRAY).call(values, numInputs);
  pthread_mutex_unlock (&PythonInterpreter::mutex);
}

void PyBehavior::getOutputs(double *values, int numOutputs) const {
  pthread_mutex_lock (&PythonInterpreter::mutex);
  behavior->method("get_outputs").pass(ONEDCARRAY).call(values, numOutputs);
  pthread_mutex_unlock (&PythonInterpreter::mutex);
}

void PyBehavior::step() {
  pthread_mutex_lock (&PythonInterpreter::mutex);
  behavior->method("step").call();
  pthread_mutex_unlock (&PythonInterpreter::mutex);
}

bool PyBehavior::canStep() const
{
  pthread_mutex_lock (&PythonInterpreter::mutex);
  bool b = behavior->method("can_step").call().returnObject()->asBool();
  pthread_mutex_unlock (&PythonInterpreter::mutex);
  return b;
}


PyBehavior* PyBehavior::fromPyObject(shared_ptr<Object> object) {
  PyBehavior* behavior = new PyBehavior(object);
  return behavior;
}

void PyBehavior::setMetaParameters(const MetaParameters& params)
{
  pthread_mutex_lock (&PythonInterpreter::mutex);
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
  pthread_mutex_unlock (&PythonInterpreter::mutex);
}

}}
