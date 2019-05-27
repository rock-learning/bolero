/**
 * @file PyLoadableBehavior.cpp
 *  @author Arne Boeckmann (arne.boeckmann@dfki.de)
 */

#include <Python.h>
#include "PyLoadableBehavior.h"
#include <stdexcept>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <iostream>


using namespace std;


namespace bolero { namespace bl_loader {

PyLoadableBehavior::PyLoadableBehavior(
        lib_manager::LibManager *libManager, const std::string &libName,
        const int libVersion)
    : LoadableBehavior(libManager, libName, libVersion)
    // numInputs and numOutputs will be set in initialize()
{}


bool PyLoadableBehavior::initialize(const std::string& initialConfigPath)
{
  std::string path = initialConfigPath;
  shared_ptr<Object> object = shared_ptr<Object>(PythonInterpreter::instance()
      .import("bolero.utils.module_loader")
      ->function("behavior_from_yaml").pass(STRING).call(&path)
      .returnObject());
  pyBehavior = shared_ptr<PyBehavior>(PyBehavior::fromPyObject(object));
  return true;
}

bool PyLoadableBehavior::configure(const std::string& configPath)
{
  std::ifstream fin(configPath.c_str());
  return configureFromIstream(fin);
}

bool PyLoadableBehavior::configureYaml(const string& yaml)
{
  std::stringstream sin(yaml);
  return configureFromIstream(sin);
}

bool PyLoadableBehavior::configureFromIstream(std::istream& stream)
{
  if(!pyBehavior)
    throw std::runtime_error("Behavior must be initialized");
  /**
   * Parse the yaml file into PyBehavior::MetaParameters
   * and call pyBehavior->setMetaParameters
   */
  vector<YAML::Node> all_docs = YAML::LoadAll(stream);
  if(all_docs.size() == 1) //we assume that there is only one document
  {
    YAML::Node doc = all_docs[0];
    PyBehavior::MetaParameters parameters;
    for(YAML::const_iterator it = doc.begin(); it != doc.end(); ++it)
    {
        string key = it->first.as<string>();
        parameters[key] = it->second.as<vector<double> >();
    }
    pyBehavior->setMetaParameters(parameters);
    return true;
  }
  else
  {
    std::cerr << "Invalid yaml document, there must be exactly 1 document." << endl;
    return false;
  }
}

void PyLoadableBehavior::init(int numInputs, int numOutputs)
{
  if(!pyBehavior)
    throw std::runtime_error("Behavior must be initialized");
  pyBehavior->init(numInputs, numOutputs);
  Behavior::init(numInputs, numOutputs);
}

void PyLoadableBehavior::setInputs(const double* values, int numInputs)
{
  if(!pyBehavior)
    throw std::runtime_error("Behavior must be initialized");
  pyBehavior->setInputs(values, numInputs);
}

void PyLoadableBehavior::getOutputs(double* values, int numOutputs) const
{
  if(!pyBehavior)
    throw std::runtime_error("Behavior must be initialized");
  pyBehavior->getOutputs(values, numOutputs);
}

void PyLoadableBehavior::step()
{
  if(!pyBehavior)
    throw std::runtime_error("Behavior must be initialized");
  pyBehavior->step();
}

bool PyLoadableBehavior::canStep() const
{
  if(!pyBehavior)
    throw std::runtime_error("Behavior must be initialized");
  return pyBehavior->canStep();
}

}}//end namespace
