#ifndef BL_COMMON_PYTHON_HELPER_H
#define BL_COMMON_PYTHON_HELPER_H

#include <string>
#include <vector>

#ifdef NO_TR1
#include <memory>
#else
#include <tr1/memory>
#endif

// forward declare PyObject
// as suggested on the python mailing list
// http://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif

namespace behavior_learning { namespace bl_loader {
/**
 * A managed pointer to PyObject which takes care about
 * memory management and reference counting.
 *
 * \note Reference counting only works if Helper::makePyObjectPtr()
 *       is used to create the pointer. Therefore you should always
 *       use Helper::makePyObjectPtr() to create new PyObjectPtrs.
 *
 * \note This type should only be used to encapsulate PyObjects that are
 *       'new references'. Wrapping a 'borrowed reference' will break Python.
 */
#ifdef NO_TR1
typedef std::shared_ptr<PyObject> PyObjectPtr;
#else
typedef std::tr1::shared_ptr<PyObject> PyObjectPtr;
#endif

typedef struct py_callable_info_s {
  PyObject *callable;
  PyObject *argTuple;
  PyObject *arg;
  size_t listSize;
} py_callable_info_t;


/**
 * A singleton that provides methods for working with Python.
 * \note The singleton initializes the Python interpreter on construction and
 *       finalizes it on destruction.
 *       There is no need to call Py_Initialize() or Py_Finalize() anywhere
 *       in the code.
 *
 * \note The singleton is constructed the first time instance() is called.
 *
 * \note This singleton is NOT thread-safe.
 *       I.e. if you want to use it from multiple threads you have to make sure,
 *       that the singelton has been initialized before starting the threads.
 *
 *       Also note that calling the python c-api from different threads is not
 *       trivial as the python interpreter is not 100% thread safe.
 *       You should read https://docs.python.org/2/c-api/init.html
 *       before trying any thread related stuff.
 */
class Helper
{
public:

  void createCallableInfo(py_callable_info_t *info, PyObject *obj,
                          const std::string &funcName, size_t listSize) const;
  void destroyCallableInfo(py_callable_info_t *info) const;
  void fillCallableInfo(py_callable_info_t *info, const double *buf,
                        size_t bufSize) const;
  void extractFromCallableInfo(const py_callable_info_t *info, double *buf,
                               size_t bufSize) const;

  /**
   * Create a new instance of a python object.
   * The instance is created by calling <name>_from_yaml() in
   * tools.python.module_loader.
   *
   * \param name The kind of object that should be created.
   *             E.g. if name="behavior"
   * \param yamlFile Path to the yaml file that will be passed on to
   *                 <name>_from_yaml().
   *                 If the string is empty no argument will be passed to
   *                 the python function.
   */
  PyObjectPtr getClassInstance(const std::string &name, const std::string& yamlFile = "") const;


  /**
   * Creates a read only 1d numpy array using the specified memory location.
   * \throws std::runtime_error if creation failed
   *
   * \param values Is not copied. The memory should remain alive until the buffer
   *               is deleted. This is the users responsibility.
   */
  PyObjectPtr create1dBuffer(const double *values, size_t size) const;

  /**
   * Creates a 1d numpy array using the specified memory location.
   * \throws std::runtime_error if creation failed
   *
   * \param values Is not copied. The memory should remain alive until the buffer
   *               is deleted. This is the users responsibility.
   */
  PyObjectPtr create1dBuffer(double *values, size_t size) const;

  /**
   * Creates a memory managed PyString that can be used
   * without worrying about reference counting.
   * \throw  std::runtime_error if the string creation fails and prints the python error to cerr.
   */
  PyObjectPtr createPyString(const std::string &str) const;

  /**
   * Imports a PyModule.
   * Python equivalent: import pyStrName
   * \throw std::runtime_error if the module creation fails and prints the python error to cerr.
   *
   * \param pyStrName The name of the module
   * \return pointer to the module
   *
   */
  PyObjectPtr importPyModule(const PyObjectPtr &pyStrName) const;

  /**
   * Retrieve an attribute named 'attrName' from object pyObj.
   * Python equivalent: pyObj.attrName
   *
   * \throw std::runtime_error in case of error.
   */
  PyObjectPtr getPyAttr(PyObjectPtr &pyObj, const std::string &attrName) const;
  PyObjectPtr getPyAttr(PyObject* pyObject, const std::string &attrName) const;
  /**
   * prints the specified object to stdout.
   */
  void printPyObject(const PyObjectPtr &pyObj) const;

  /**
   * Tries to invoke the specified method on the object without parameters
   * \throws std::runtime_error if the call fails
   */
  PyObjectPtr callPyMethod(const PyObjectPtr &pyObj, const std::string &funcName) const;

  void printPyTraceback() const;
  bool checkPyError() const;

  /**
   * Creates a PyObjectPtr with the correct deleter.
   * You should always (!!!) use this method to create PyObjectPtrs.
   * Otherwise they will not be deleted correctly.
   */
  static PyObjectPtr makePyObjectPtr(PyObject* p);

  /**
   * \return true if the object can be evaluated to true, false otherwise.
   * \throws std::runtime_error of object cannot be evaluate as boolean.
   */
  bool isPyObjectTrue(const PyObjectPtr &obj) const;

  /**
   * Returns a new PyList
   */
  PyObjectPtr createPyList(const int size) const;


  //FIXME this could be solved using templates or overloaded.
  //      But the semantics of the different overloads is slightly different,
  //      maybe overloading is not such a good idea?
  /**
   * Returns a new PyList containing copies of the specified strings
   */
  PyObjectPtr createPyListFromStrings(const std::vector<std::string>& strings) const;
  /**
   * Returns a new PyList containing copies of the specified doubles
   */
  PyObjectPtr createPyListFromDoubles(const std::vector<double>& values) const;

  /*
  * Returns a new PyList containing references to the specified objects.
  */
  PyObjectPtr createPyListFromObjects(const std::vector<PyObjectPtr>& values) const;

  /**
   * Adds the specified value to the python module path
   * \throws std::runtime_error
   */
  void addToPyModulePath(const std::string &value) const;

  /**
   * \return the singleton instance of the Helper.
   */
  static const Helper& instance();

private:

  /**
   * Should only be called from instance().
   * Ensures that all necessary python functionality is loaded (e.g. numpy arrays).
   * \throws std::runtime_error if the python environment has not been initialized.
   */
  Helper();

  /**
   * Finalizes the python interpreter
   */
  ~Helper();

  Helper(Helper const&);              // Don't Implement
  void operator=(Helper const&); // Don't implement

   /* If True the Helper will call Py_Finalize() upon destruction.
    * This is only true if the Helper has also initialized Python.
    * If Python has been initialized by someone else we assume that
    * someone else is also responsible for finalizing it. */
  bool finalizePython;

};

}}

#endif /* BL_COMMON_PYTHON_HELPER_H */
