#include "QuaternionDmpConfig.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h>
#include <yaml-cpp/yaml.h>

using namespace std;
namespace dmp
{

QuaternionDmpConfig::QuaternionDmpConfig() : fullyInitialized(false)
{}

bool QuaternionDmpConfig::fromYamlFile(const std::string &filepath, const std::string &name)
{
  ifstream fin(filepath.c_str());
  return fromIstream(fin, name);
}

bool QuaternionDmpConfig::fromYamlString(const std::string &yaml, const std::string &name)
{
  stringstream sin(yaml);
  return fromIstream(sin, name);
}

bool QuaternionDmpConfig::fromIstream(std::istream& stream, std::string name)
{
  vector<YAML::Node> all_docs = YAML::LoadAll(stream);
  for(size_t i = 0; i<all_docs.size(); i++) {
    YAML::Node doc = all_docs[i];
    string name_buf;
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

    config_name = name_buf;

    if(doc["executionTime"])
      executionTime = doc["executionTime"].as<double>();
    else
      new_config_is_complete = false;

    if(doc["startPosition"])
      startPosition = doc["startPosition"].as<std::vector<double> >();
    else
      new_config_is_complete = false;

    if(doc["endPosition"])
      endPosition = doc["endPosition"].as<std::vector<double> >();
    else
      new_config_is_complete = false;

    if(doc["startVelocity"])
      startVelocity = doc["startVelocity"].as<std::vector<double> >();
    else
      new_config_is_complete = false;

    fullyInitialized = new_config_is_complete || has_been_initialized;

    return isValid();
  }
  return false;
}

bool QuaternionDmpConfig::isValid() const
{
  if(executionTime <= 0.0)
  {
    cerr << "QuaternionDmpConfing invalid. Execution time should be > 0." << endl;
    return false;
  }
  if(startPosition.size() != 4)
  {
    cerr << startPosition.size() << endl;
    cerr << "QuaternionDmpConfing invalid. Start position should have exactly 4 elements" << endl;
    return false;
  }
  if(endPosition.size() != 4)
  {
    cerr << "QuaternionDmpConfing invalid. End position should have exactly 4 elements" << endl;
    return false;
  }
  if(startVelocity.size() != 3)
  {
    cerr << "QuaternionDmpConfing invalid. Start velocity should have exactly 3 elements" << endl;
    return false;
  }

  for(int i = 0; i < 4; ++i)
  {
    if(std::isnan(startPosition[i]))
    {
      cerr << "QuaternionDmpConfing invalid. Start position contains NaN" << endl;
      return false;
    }
    if(std::isnan(endPosition[i]))
    {
      cerr << "QuaternionDmpConfing invalid. End position contains NaN" << endl;
      return false;
    }
    if(std::isinf(startPosition[i]))
    {
      cerr << "QuaternionDmpConfing invalid. Start position contains inf" << endl;
      return false;
    }
    if(std::isinf(endPosition[i]))
    {
      cerr << "QuaternionDmpConfing invalid. End position contains inf" << endl;
      return false;
    }
  }

  for(int i = 0; i < 3; ++i)
  {
    if(std::isnan(startVelocity[i]))
    {
      cerr << "QuaternionDmpConfing invalid. Start velocity contains NaN" << endl;
      return false;
    }
    if(std::isinf(startVelocity[i]))
    {
      cerr << "QuaternionDmpConfing invalid. Start velocity contains inf" << endl;
      return false;
    }
  }
  return fullyInitialized;
}

void QuaternionDmpConfig::toYamlFile(std::string filepath)
{
  assert(filepath.size() > 0);
  YAML::Emitter out;
  out << YAML::BeginDoc;
  out << YAML::BeginMap << YAML::Key << "name" << YAML::Value << config_name;
  out << YAML::Key << "executionTime" << YAML::Value << executionTime;
  out << YAML::Key << "startPosition" << YAML::Value << YAML::Flow << startPosition;
  out << YAML::Key << "endPosition" << YAML::Value << YAML::Flow << endPosition;
  out << YAML::Key << "startVelocity" << YAML::Value << YAML::Flow << startVelocity;
  out << YAML::EndMap;
  out << YAML::EndDoc;

  ofstream fout(filepath.c_str(), ios::out | ios::app);
  assert(fout.is_open());
  fout << out.c_str();
  fout.close();
}
}
