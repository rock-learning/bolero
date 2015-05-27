#include <Python.h> //needs to be include before anything else to avoid stupid warnings
#include <numpy/arrayobject.h>

#include "Helper.h"
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
#include <cstdio>

#ifdef NO_TR1
  using namespace std;
#else
  using namespace std::tr1;
#endif

namespace behavior_learning { namespace bl_loader {
/**
 * This deleter should be used when managing PyObject with
 * std::tr1::shared_ptr (std::shared_ptr in c++11)
 *
 */
struct PyObjectDeleter
{
  void operator()(PyObject* p) const {
    Py_XDECREF(p);
    }
};

PyObjectPtr Helper::makePyObjectPtr(PyObject* p)
{
  return PyObjectPtr(p, PyObjectDeleter());
}

Helper::Helper() : finalizePython(false)
{
  if(!Py_IsInitialized())
  {
    Py_Initialize();
    finalizePython = true;
  }
  import_array();
}

Helper::~Helper()
{
  if(finalizePython && Py_IsInitialized())
  {
    Py_Finalize();
  }
}

const Helper& Helper::instance()
{
  static Helper h;
  return h;
}

PyObjectPtr Helper::createPyString(const std::string &str) const
{
  PyObjectPtr pyStr = makePyObjectPtr(PyString_FromString(str.c_str()));
  if(!pyStr)
  {
    PyErr_Print();
    throw std::runtime_error("unable to create PyString");
  }
  return pyStr;
}

PyObjectPtr Helper::importPyModule(const PyObjectPtr &pyStrName) const
{
  PyObjectPtr pyModule = makePyObjectPtr(PyImport_Import(pyStrName.get()));
  if(!pyModule)
  {
    PyErr_Print();
    throw std::runtime_error("unable to load module");
  }
  return pyModule;
}

PyObjectPtr Helper::getPyAttr(PyObjectPtr &pyObj, const std::string &attrName) const
{
  return getPyAttr(pyObj.get(), attrName);
}

PyObjectPtr Helper::getPyAttr(PyObject* pyObject, const std::string &attrName) const
{
  PyObjectPtr pyAttr = makePyObjectPtr(PyObject_GetAttrString(pyObject, attrName.c_str()));

  if(!pyAttr)
  {
    PyErr_Print();
    throw std::runtime_error("unable to load python attribute");
  }
  return pyAttr;
}


void Helper::printPyObject(const PyObjectPtr &pyObj) const
{
  if(pyObj)
  {
    if(-1 == PyObject_Print(pyObj.get(), stdout, Py_PRINT_RAW))
    {
      throw std::runtime_error("unable to print PyObject");
    }
  }
  else
  {
    throw std::runtime_error("unable to print PyObject. Pointer is NULL.");
  }
}

PyObjectPtr Helper::callPyMethod(const PyObjectPtr &pyObj, const std::string &funcName) const
{
  if(funcName.empty())
  {
    throw std::runtime_error("method name not valid");
  }
  if(pyObj)
  {
    PyObject* result = PyObject_CallMethod(pyObj.get(), const_cast<char*>(funcName.c_str()), NULL);
    if(!result)
    {
      PyErr_Print();
      throw std::runtime_error("calling method " + funcName + " failed");
    }
    return makePyObjectPtr(result);
  }
  else
  {
    throw std::runtime_error("parameter is null");
  }
}

void Helper::createCallableInfo(py_callable_info_t *info, PyObject *obj,
                        const std::string &funcName, size_t listSize) const {
  // get the callable
  info->callable = PyObject_GetAttrString(obj, (char*)funcName.c_str());
  if(!info->callable) {
    std::string errString = "Optimizer has no method \"" + funcName + "\"";
    throw std::runtime_error(errString);
  }
  if(!PyCallable_Check(info->callable)) {
    std::string errString = "\"" + funcName + "\" is not callable";
    throw std::runtime_error(errString);
  }

  // create the argument
  info->arg = PyList_New(listSize);
  if(!info->arg) {
    std::stringstream s;
    s << "Could not create Python list of size " << listSize;
    throw std::runtime_error(s.str());
  }
  for(size_t i = 0; i < listSize; ++i) {
    Py_INCREF(Py_None);
    PyList_SetItem(info->arg, i, Py_None);
  }
  info->listSize = listSize;

  // pack argument into tuple
  info->argTuple = PyTuple_Pack(1, info->arg);
  if(!info->argTuple) {
    std::string errString = "Could not create Python tuple!";
    throw std::runtime_error(errString);
  }
}

void Helper::destroyCallableInfo(py_callable_info_t *info) const {
  Py_DECREF(info->argTuple);
  Py_DECREF(info->arg);
  Py_DECREF(info->callable);
  info->argTuple = info->arg = info->callable = NULL;
  info->listSize = 0;
}

void Helper::fillCallableInfo(py_callable_info_t *info, const double *buf,
                      size_t bufSize) const {
  PyObject *pItem;
  // fill the existing list
  for(size_t i = 0; i < std::min(info->listSize, bufSize); ++i) {
    pItem = PyFloat_FromDouble(buf[i]);
    if(!pItem) {
      throw std::runtime_error("Could not create PyFloat from double");
    }
    // SetItem steals the reference so we do not need to DECREF
    PyList_SetItem(info->arg, i, pItem);
  }
  // expand the existing list if needed
  for(size_t i = std::min(info->listSize, bufSize); i < bufSize; ++i) {
    pItem = PyFloat_FromDouble(buf[i]);
    if(!pItem) {
      throw std::runtime_error("Could not create PyFloat from double");
    }
    PyList_Append(info->arg, pItem);
    // Append does not steal the reference so we need to DECREF
    Py_DECREF(pItem);
  }
  // shrink the list when needed
  if(bufSize < info->listSize) {
    PyList_SetSlice(info->arg, bufSize, info->listSize, NULL);
  }
  info->listSize = bufSize;
}

void Helper::extractFromCallableInfo(const py_callable_info_t *info, double *buf,
                             size_t bufSize) const {
  PyObject *pItem;
  assert(info->listSize == bufSize);
  for(size_t i = 0; i < bufSize; ++i) {
    pItem = PyList_GetItem(info->arg, i);
    if(!pItem) {
      PyErr_Print();
      fprintf(stderr, "%lu %lu\n", i, info->listSize);
      throw std::runtime_error("Could not get item from PyList");
    }
    if(!PyFloat_Check(pItem)) {
      if(PyNumber_Check(pItem))
    	throw std::runtime_error("List item is a number but must be a float");
      else
        throw std::runtime_error("List item is neither a float nor another type of number");
    }
    buf[i] = PyFloat_AsDouble(pItem);
    if(PyErr_Occurred()) {
      throw std::runtime_error("Conversion to double failed");
    }
  }
}


PyObjectPtr Helper::getClassInstance(const std::string &name, const std::string& yamlFile) const {

  const std::string moduleName = "tools.python.module_loader";
  std::string funcName = name + "_from_yaml";

  PyObjectPtr pyModuleString = createPyString(moduleName);
  PyObjectPtr pyModule = importPyModule(pyModuleString);
  PyObjectPtr pyFunc = getPyAttr(pyModule, funcName);

  PyObject *pyInstance = NULL;
  //call function either with arguments or without depending on yamlFile
  if(yamlFile.empty()) {
    pyInstance = PyEval_CallObject(pyFunc.get(), NULL);
  }
  else {
    //note: the parentheses around 's' in the format string
    //are necessary to create a tuple with only one element
    pyInstance = PyEval_CallFunction(pyFunc.get(), "(s)", yamlFile.c_str());
  }

  if(!pyInstance) {
    // FIXME: we want to print the python error when we are living in C++
    //   but printing the PyError when we live in Python will clear the error flag
    PyErr_Print();
    throw std::runtime_error("Error calling " + funcName);
  }
  return makePyObjectPtr(pyInstance);
}

static inline PyObject* __create1dBuffer(double *values, int size, bool isReadOnly) {

  //Python expects sizeof(double) == 8.
  typedef int static_assert_sizeof_double_is_8[sizeof(double) == 8 ? 1 : -1];

  npy_intp dims[1] = {size};
  //FIXME not sure if dims should be stack allocated
  return PyArray_SimpleNewFromData(1, &dims[0], NPY_DOUBLE, (void*) values);
}

PyObjectPtr Helper::create1dBuffer(const double *values, size_t size) const {
  return create1dBuffer(const_cast<double*>(values), size);
}
PyObjectPtr Helper::create1dBuffer(double *values, size_t size) const {
  PyObject* pyBuffer = __create1dBuffer(values, size, true);
  if(!pyBuffer)
  {
    PyErr_Print();
    throw std::runtime_error("Error while creating 1d buffer");
  }
  return makePyObjectPtr(pyBuffer);
}

void Helper::printPyTraceback() const {
  PyObject *type, *value, *traceback;
  PyObject *tracebackModule;
  char *chrRetval;
  
  PyErr_Fetch(&type, &value, &traceback);
  
  tracebackModule = PyImport_ImportModule("traceback");
  if (tracebackModule != NULL) {
    PyObject *tbList, *emptyString, *strRetval;
    
    tbList = PyObject_CallMethod(tracebackModule,
                                 (char*)"format_exception",
                                 (char*)"OOO",
                                 type,
                                 value == NULL ? Py_None : value,
                                 traceback == NULL ? Py_None : traceback);
    
    emptyString = PyString_FromString("");
    strRetval = PyObject_CallMethod(emptyString, (char*)"join",
                                    (char*)"O", tbList);
    
    chrRetval = strdup(PyString_AsString(strRetval));
    
    Py_DECREF(tbList);
    Py_DECREF(emptyString);
    Py_DECREF(strRetval);
    Py_DECREF(tracebackModule);
  } else {
    chrRetval = strdup("Unable to import traceback module.");
  }

  Py_DECREF(type);
  Py_XDECREF(value);
  Py_XDECREF(traceback);
  
  fprintf(stderr, "%s", chrRetval);
  free(chrRetval);
}

bool Helper::checkPyError() const {
  if(!PyErr_Occurred()) {
    return false;
  }
  printPyTraceback();
  return true;
}

bool Helper::isPyObjectTrue(const PyObjectPtr &obj) const
{
  assert(obj);
  const int ret = PyObject_IsTrue(obj.get());
  switch(ret)
  {
  case 0:
    return false;
  case 1:
    return true;
  case -1:
    //PyObject_IsTrue returns -1 for objects that cannot be evaluated as bool
    throw std::runtime_error("The return value of canStep() could not be evaluated as bool");
  default:
    throw std::runtime_error("PyObject_IsTrue() returned an unknown error");
  }
}

PyObjectPtr Helper::createPyList(const int size) const
{
  PyObject* list = PyList_New(size);
  if(!list)
  {
    PyErr_Print();
    throw std::runtime_error("Error while creating list");
  }
  return makePyObjectPtr(list);
}

PyObjectPtr Helper::createPyListFromStrings(const std::vector<std::string>& strings) const
{
  PyObjectPtr list(createPyList(strings.size()));
  for(size_t i = 0; i < strings.size(); ++i)
  {
    PyObject* str = PyString_FromString(strings[i].c_str());
    if(!str)
    {
      PyErr_Print();
      throw std::runtime_error("Error while creating list");
    }
    PyList_SetItem(list.get(), i, str);
    //PyList_SetItem steals the string reference, therefore we do not decref it
  }
  return list;
}

PyObjectPtr Helper::createPyListFromDoubles(const std::vector<double>& values) const
{
  PyObjectPtr list(createPyList(values.size()));
  for(size_t i = 0; i < values.size(); ++i)
  {
    PyObject* pyVal =  PyFloat_FromDouble(values[i]);
    if(!pyVal)
    {
      PyErr_Print();
      throw std::runtime_error("Error while creating list");
    }
    PyList_SetItem(list.get(), i, pyVal);
    //PyList_SetItem steals the reference, therefore we do not decref it
  }
  return list;
}

PyObjectPtr Helper::createPyListFromObjects(const std::vector<PyObjectPtr>& values) const
{
  PyObjectPtr list(createPyList(values.size()));
  for(size_t i = 0; i < values.size(); ++i)
  {
    PyObject* pyVal =  values[i].get();
    if(!pyVal)
    {
      PyErr_Print();
      throw std::runtime_error("Error while creating list");
    }
    PyList_SetItem(list.get(), i, pyVal);
    //PyList_SetItem steals the reference from the PyObjectPtr.
    //But it is impossible to steal from the PyObjectPtr, because it always
    //decrefs on destruction. We need to incref here to allow for the decref
    //in PyObjectPtr
    Py_INCREF(pyVal);
  }
  return list;
}

void Helper::addToPyModulePath(const std::string &value) const
{
  PyObject *pModule = NULL;
  pModule = PyImport_ImportModule("sys");
  if(pModule) {
    PyObjectPtr pPath = getPyAttr(pModule, "path");
    if(PyList_Check(pPath.get())) {
      PyObjectPtr pString = createPyString(value);
      int res = PyList_Append(pPath.get(), pString.get());
      if(res != 0) {
        PyErr_Print();
        throw std::runtime_error("could not append to sys.path");
      }
    } else {
      PyErr_Print();
      throw std::runtime_error("'sys.path' is not a list");
    }
  } else {
    PyErr_Print();
    throw std::runtime_error(" Failed to load 'sys'");
  }
  Py_XDECREF(pModule);
}



}}