#include <yaml-cpp/parser.h>
#include "RigidBodyDmpConfig.h"
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace dmp
{
using std::ifstream;
using std::stringstream;
using std::string;
using std::clog;
using std::ofstream;
using std::ios;
using std::endl;
using std::vector;


RigidBodyDmpConfig::RigidBodyDmpConfig()
  : fullyInitialized(false)
{
}

bool RigidBodyDmpConfig::isValid() const
{
  return translationConfig.is_valid() && rotationConfig.isValid();
}

bool RigidBodyDmpConfig::fromYamlFile(const std::string &filepath, const std::string &name)
{
  ifstream fin(filepath.c_str());
  return fromIstream(fin, name);
}

bool RigidBodyDmpConfig::fromYamlString(const std::string &yaml, const std::string &name)
{
  stringstream sin(yaml);
  return fromIstream(sin, name);
}

bool RigidBodyDmpConfig::fromIstream(std::istream& stream, std::string name)
{
  YAML::Node doc;
  string name_buf;
  vector<YAML::Node> all_docs = YAML::LoadAll(stream);
  for(size_t i = 0; i<all_docs.size(); i++) {
    YAML::Node doc = all_docs[i];
    if(doc["name"])
    {
      name_buf = doc["name"].as<std::string>();
      if(name == "") {
        name = name_buf;
      }
      if(name_buf != name) {
        continue;
      }
    }

    const bool has_been_initialized = fullyInitialized;
    bool new_config_is_complete = true;

    rotationConfig.config_name = name_buf;
    translationConfig.config_name = name_buf;

    if(doc["startPosition"])
      translationConfig.dmp_startPosition = doc["startPosition"].as<std::vector<double> >();
    else
      new_config_is_complete = false;

    if(doc["endPosition"])
      translationConfig.dmp_endPosition = doc["endPosition"].as<std::vector<double> >();
    else
      new_config_is_complete = false;

    if(doc["startVelocity"])
      translationConfig.dmp_startVelocity = doc["startVelocity"].as<std::vector<double> >();
    else
      new_config_is_complete = false;

    if(doc["endVelocity"])
      translationConfig.dmp_endVelocity = doc["endVelocity"].as<std::vector<double> >();
    else
      new_config_is_complete = false;

    if(doc["startAcceleration"])
      translationConfig.dmp_startAcceleration = doc["startAcceleration"].as<std::vector<double> >();
    else
      new_config_is_complete = false;

    if(doc["endAcceleration"])
      translationConfig.dmp_endAcceleration = doc["endAcceleration"].as<std::vector<double> >();
    else
      new_config_is_complete = false;

    if(doc["startRotation"])
      rotationConfig.startPosition = doc["startRotation"].as<std::vector<double> >();
    else
      new_config_is_complete = false;

    if(doc["endRotation"])
      rotationConfig.endPosition = doc["endRotation"].as<std::vector<double> >();
    else
      new_config_is_complete = false;

    if(doc["startAngularVelocity"])
      rotationConfig.startVelocity = doc["startAngularVelocity"].as<std::vector<double> >();
    else
      new_config_is_complete = false;

    if(doc["executionTime"])
    {
      double executionTime = doc["executionTime"].as<double>();
      translationConfig.dmp_execution_time = executionTime;
      rotationConfig.executionTime = executionTime;
    }
    else
      new_config_is_complete = false;

    fullyInitialized = new_config_is_complete || has_been_initialized;

    rotationConfig.fullyInitialized = fullyInitialized;
    translationConfig.fullyInitialized = fullyInitialized;

    return isValid();
  }
  return false;
}

void RigidBodyDmpConfig::toYamlFile(std::string filepath)
{

  assert(filepath.size() > 0);
  YAML::Emitter out;
  out << YAML::BeginDoc;
  out << YAML::BeginMap << YAML::Key << "name" << YAML::Value << translationConfig.config_name;
  out << YAML::Key << "startPosition" << YAML::Value << YAML::Flow << translationConfig.dmp_startPosition;
  out << YAML::Key << "endPosition" << YAML::Value << YAML::Flow << translationConfig.dmp_endPosition;
  out << YAML::Key << "startVelocity" << YAML::Value << YAML::Flow << translationConfig.dmp_startVelocity;
  out << YAML::Key << "endVelocity" << YAML::Value << YAML::Flow << translationConfig.dmp_endVelocity;
  out << YAML::Key << "startAcceleration" << YAML::Value << YAML::Flow << translationConfig.dmp_startAcceleration;
  out << YAML::Key << "endAcceleration" << YAML::Value << YAML::Flow << translationConfig.dmp_endAcceleration;
  out << YAML::Key << "startRotation" << YAML::Value << YAML::Flow << rotationConfig.startPosition;
  out << YAML::Key << "endRotation" << YAML::Value << YAML::Flow << rotationConfig.endPosition;
  out << YAML::Key << "startAngularVelocity" << YAML::Value << YAML::Flow << rotationConfig.startVelocity;
  out << YAML::Key << "executionTime" << YAML::Value << translationConfig.dmp_execution_time;
  out << YAML::EndMap;
  out << YAML::EndDoc;

  ofstream fout(filepath.c_str(), ios::out | ios::app);
  assert(fout.is_open());
  fout << out.c_str();
  fout.close();
}
}
