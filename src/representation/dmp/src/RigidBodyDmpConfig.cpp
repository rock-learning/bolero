#include <yaml-cpp/parser.h>
#include "RigidBodyDmpConfig.h"
#include <sstream>
#include <fstream>
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

bool RigidBodyDmpConfig::isValid() const
{
  return translationConfig.is_valid() && rotationConfig.isValid();
}

bool RigidBodyDmpConfig::fromYamlFile(const std::string &filepath, const std::string &name)
{
  ifstream fin(filepath.c_str());
  YAML::Parser parser(fin);
  return fromYamlParser(parser, name);
}

bool RigidBodyDmpConfig::fromYamlString(const std::string &yaml, const std::string &name)
{
  stringstream sin(yaml);
  YAML::Parser parser(sin);
  return fromYamlParser(parser, name);
}

bool RigidBodyDmpConfig::fromYamlParser(YAML::Parser &parser, std::string name)
{
  YAML::Node doc;
  string name_buf;
  while(parser.GetNextDocument(doc))
  {
    doc["name"] >> name_buf;
    if(name == "")
    {
      name = name_buf;
    }
    if(name_buf != name)
    {
      continue;
    }
    else
    {

      fullyInitialized = true;
      rotationConfig.config_name = name_buf;
      translationConfig.config_name = name_buf;

      if(doc.FindValue("startPosition"))
        doc["startPosition"] >> translationConfig.dmp_startPosition;
      else
        fullyInitialized = false;

      if(doc.FindValue("endPosition"))
        doc["endPosition"] >> translationConfig.dmp_endPosition;
      else
        fullyInitialized = false;

      if(doc.FindValue("startVelocity"))
        doc["startVelocity"] >> translationConfig.dmp_startVelocity;
      else
        fullyInitialized = false;

      if(doc.FindValue("endVelocity"))
        doc["endVelocity"] >> translationConfig.dmp_endVelocity;
      else
        fullyInitialized = false;

      if(doc.FindValue("startAcceleration"))
        doc["startAcceleration"] >> translationConfig.dmp_startAcceleration;
      else
        fullyInitialized = false;

      if(doc.FindValue("endAcceleration"))
        doc["endAcceleration"] >> translationConfig.dmp_endAcceleration;
      else
        fullyInitialized = false;

      if(doc.FindValue("startRotation"))
        doc["startRotation"] >> rotationConfig.startPosition;
      else
        fullyInitialized = false;

      if(doc.FindValue("endRotation"))
        doc["endRotation"] >> rotationConfig.endPosition;
      else
        fullyInitialized = false;

      if(doc.FindValue("startAngularVelocity"))
        doc["startAngularVelocity"] >> rotationConfig.startVelocity;
      else
        fullyInitialized = false;

      if(doc.FindValue("executionTime"))
      {
        double executionTime;
        doc["executionTime"] >> executionTime;
        translationConfig.dmp_execution_time = executionTime;
        rotationConfig.executionTime = executionTime;
      }
      else
        fullyInitialized = false;

      rotationConfig.fullyInitialized = fullyInitialized;
      translationConfig.fully_initialized = fullyInitialized;

      return isValid();
    }
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
