/**
 * @file PyLoadableBehavior.h
 *  @author Arne Boeckmann (arne.boeckmann@dfki.de)
 */

#pragma once
#include "PyBehavior.h"
#include <string>
#include <Behavior.h>
#include <memory>
#include "Helper.h"
#include <LoadableBehavior.h>
#include "PyLoadable.h"

// forward declare PyObject
// as suggested on the python mailing list
// http://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif

namespace YAML {
  class Parser;
}

namespace behavior_learning { namespace bl_loader {

/**
 * Loads python behaviors and wraps them.
 */
class PyLoadableBehavior : public LoadableBehavior, public PyLoadable {
public:

  PyLoadableBehavior(lib_manager::LibManager *libManager,
              const std::string &libName, const int libVersion);

  /**
   * Creates the behavior using module_loader.behavior_from_yaml()
   *
   * \param initialConfigFile should be a yaml file that can be passed
   *                          to the python function
   *                          module_loader.behavior_from_yaml()
   *                          If this string is empty no parameter
   *                          will be passed to behavior_from_yaml()
   *                          and the default will be used.
   */
  virtual bool initialize(const std::string& initialConfigPath);

  /**
   * Calls set_meta_parameters() on the wrapped python behavior.
   *
   * \param configPath Path to a yaml file containing the meta parameters
   *                   as key value pairs.
   *                   Keys may be strings.
   *                   Values have to be lists of doubles.
   *
   *                   Example config file:
   *                   > asd : [0.0, 0.1, 0.2, 0.3, 42.0, 42.1, 42.2]
   *                   > aaa : [0.4, 0.41, 0.4, 44.43, 4442.0, 442.1, 42.2]
   *                   > singleValue : [42.0]
   */
  virtual bool configure(const std::string& configPath);

  /**
   * Calls set_meta_parameters() on the wrapped python behavior.
   * For now only double arrays are allowed as values.
   */
  virtual bool configureYaml(const std::string& yaml);


  virtual void setInputs(const double *values, int numInputs);
  virtual void getOutputs(double *values, int numOutputs) const;

  virtual void step();
  virtual bool canStep() const;

private:

  bool configureFromYamlParser(YAML::Parser& parser);

  /**Wrapper arround py_behavior that implements the 'old' Behavior interface */
  std::auto_ptr<PyBehavior> pyBehavior;
  /**Handle to the python behavior. The memory is managed by python */
  PyObjectPtr py_behavior;


};
}}//end namespace
