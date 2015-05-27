#include <Python.h>
#include "PyBehavior.h"
#include <cassert>
#include <stdexcept>

using namespace std;

namespace behavior_learning { namespace bl_loader {

PyBehavior::PyBehavior(int numInputs, int numOutputs)
  : Behavior(numInputs, numOutputs)
{}

PyBehavior::~PyBehavior() {
  Helper::instance().destroyCallableInfo(&set_inputs);
  Helper::instance().destroyCallableInfo(&get_outputs);
}

#ifdef USE_MEMORYVIEWS

void PyBehavior::setInputs(const double *values, int numInputs) {
  assert(py_behavior);
  PyObjectPtr memView = Helper::instance().create1dBuffer(values, numInputs);
  if(memView) {
    PyObject *pResult = PyObject_CallMethod(py_behavior.get(), (char*)"set_inputs",
                                            (char*)"O", memView.get());
    Py_XDECREF(pResult);
  }
  if(PyErr_Occurred()) {
    PyErr_Print();
  }
}

void PyBehavior::getOutputs(double *values, int numOutputs) const {
  assert(py_behavior);
  PyObjectPtr memView = Helper::instance().create1dBuffer(values, numOutputs);
  PyObject *pResult = PyObject_CallMethod(py_behavior.get(), (char*)"get_outputs",
                                          (char*)"O", memView.get());
  Py_XDECREF(pResult);

  if(PyErr_Occurred()) {
    PyErr_Print();
  }
}
#else // USE_MEMORYVIEWS

void PyBehavior::setInputs(const double *values, int numInputs) {
  Helper::instance().fillCallableInfo(&set_inputs, values, numInputs);
  PyObject *pResult = PyObject_CallObject(set_inputs.callable,
                                          set_inputs.argTuple);
  if(pResult) {
    Py_DECREF(pResult);
  } else {
    PyErr_Print();
  }
}

void PyBehavior::getOutputs(double *values, int numOutputs) const {
  PyObject *pResult = PyObject_CallObject(get_outputs.callable,
                                          get_outputs.argTuple);
  if(pResult) {
    Helper::instance().extractFromCallableInfo(&get_outputs, values, numOutputs);
    Py_DECREF(pResult);
  } else {
    PyErr_Print();
  }
}

#endif // USE_MEMORYVIEWS

void PyBehavior::step() {
  assert(py_behavior);
  PyObject *pResult = PyObject_CallMethod(py_behavior.get(), (char*)"step", NULL);
  if(!pResult) {
    PyErr_Print();
  }
  Py_XDECREF(pResult);
}

bool PyBehavior::canStep() const
{
  PyObjectPtr result = Helper::instance().callPyMethod(py_behavior, "can_step");
  return Helper::instance().isPyObjectTrue(result);
}


PyBehavior* PyBehavior::fromPyObject(PyObject *pObj) {
  assert(pObj);
  PyObjectPtr numInputsResult = Helper::instance().getPyAttr(pObj, "num_inputs");

  assert(PyInt_Check(numInputsResult.get()));
  long numInputs = PyInt_AsLong(numInputsResult.get());

  PyObjectPtr numOutputsResult = Helper::instance().getPyAttr(pObj, "num_outputs");

  assert(PyInt_Check(numOutputsResult.get()));
  long numOutputs = PyInt_AsLong(numOutputsResult.get());

  PyBehavior *behavior = new PyBehavior(numInputs, numOutputs);
  behavior->py_behavior = Helper::instance().makePyObjectPtr(pObj);
  Helper::instance().createCallableInfo(&behavior->set_inputs, pObj, "set_inputs", numInputs);
  Helper::instance().createCallableInfo(&behavior->get_outputs, pObj, "get_outputs", numOutputs);
  return behavior;
}

void PyBehavior::setMetaParameters(const MetaParameters &params)
{
  vector<string> keys;
  vector<PyObjectPtr> values;
  MetaParameters::const_iterator it;
  for(it = params.begin(); it != params.end(); ++it)
  {
    keys.push_back(it->first);
    PyObjectPtr pyVal = Helper::instance().createPyListFromDoubles(it->second);
    values.push_back(pyVal);
  }

  PyObjectPtr pyKeys = Helper::instance().createPyListFromStrings(keys);
  //create a list of lists
  PyObjectPtr pyValues = Helper::instance().createPyListFromObjects(values);
  PyObjectPtr methodName = Helper::instance().createPyString("set_meta_parameters");
  PyObject* pyResult = PyObject_CallMethodObjArgs(py_behavior.get(), methodName.get(),
                                                  pyKeys.get(), pyValues.get(), NULL);
  if(!pyResult)
  {
    PyErr_Print();
    throw std::runtime_error("Error while calling set_meta_parameters");
  }
  Py_XDECREF(pyResult);
}

}}