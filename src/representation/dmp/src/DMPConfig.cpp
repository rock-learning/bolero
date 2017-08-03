#include "DMPConfig.h"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <math.h>
#include <sstream>
#include <iostream>

using namespace std;
namespace dmp_cpp
{

DMPConfig::DMPConfig() : config_name("NOT INITIALIZED"), dmp_execution_time(0.0), fullyInitialized(false)
{}

DMPConfig::DMPConfig(const string& filepath, const string& name) :
    config_name("NOT INITIALIZED"), dmp_execution_time(0.0), fullyInitialized(false)
{
  if(!from_yaml_file(filepath, name))
  {
    stringstream ss;
    ss << "DMPModel: Unable to load dmp config file: " << filepath;
    throw std::runtime_error(ss.str());
  }
}

bool DMPConfig::from_yaml_string(const string& yaml, const string& name)
{
  stringstream sin(yaml);
  return from_yaml_istream(sin, name);
}

bool DMPConfig::from_yaml_file(const string& filepath, const string& name){
  ifstream fin(filepath.c_str());
  return from_yaml_istream(fin, name);
}

bool DMPConfig::from_yaml_istream(std::istream& stream, string name)
{
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

      config_name = name_buf;

      if(doc["dmp_execution_time"])
        dmp_execution_time = doc["dmp_execution_time"].as<double>();
      else
        new_config_is_complete = false;

      if(doc["dmp_startPosition"])
        dmp_startPosition = doc["dmp_startPosition"].as<std::vector<double> >();
      else
        new_config_is_complete = false;

      if(doc["dmp_endPosition"])
        dmp_endPosition = doc["dmp_endPosition"].as<std::vector<double> >();
      else
        new_config_is_complete = false;

      if(doc["dmp_startVelocity"])
        dmp_startVelocity = doc["dmp_startVelocity"].as<std::vector<double> >();
      else
        new_config_is_complete = false;

      if(doc["dmp_endVelocity"])
        dmp_endVelocity = doc["dmp_endVelocity"].as<std::vector<double> >();
      else
        new_config_is_complete = false;

      if(doc["dmp_startAcceleration"])
        dmp_startAcceleration = doc["dmp_startAcceleration"].as<std::vector<double> >();
      else
        new_config_is_complete = false;

      if(doc["dmp_endAcceleration"])
        dmp_endAcceleration = doc["dmp_endAcceleration"].as<std::vector<double> >();
      else
        new_config_is_complete = false;

      fullyInitialized = new_config_is_complete || has_been_initialized;

      return is_valid();
  }
  return false;
}

void DMPConfig::to_yaml_file(string filepath){
  YAML::Emitter out;
  //out.SetDoublePrecision(15);
  out << YAML::BeginDoc;
  out << YAML::BeginMap << YAML::Key << "name" << YAML::Value << config_name;
  out << YAML::Key << "dmp_execution_time" << YAML::Value << dmp_execution_time;
  out << YAML::Key << "dmp_startPosition" << YAML::Value << YAML::Flow << dmp_startPosition;
  out << YAML::Key << "dmp_endPosition" << YAML::Value << YAML::Flow << dmp_endPosition;
  out << YAML::Key << "dmp_startVelocity" << YAML::Value << YAML::Flow << dmp_startVelocity;
  out << YAML::Key << "dmp_endVelocity" << YAML::Value << YAML::Flow << dmp_endVelocity;
  out << YAML::Key << "dmp_startAcceleration" << YAML::Value << YAML::Flow << dmp_startAcceleration;
  out << YAML::Key << "dmp_endAcceleration" << YAML::Value << YAML::Flow << dmp_endAcceleration;
  out << YAML::EndMap;
  out << YAML::EndDoc;

  ofstream fout(filepath.c_str(), ios::out | ios::app );
  fout << out.c_str();
  fout.close();
}

bool DMPConfig::is_valid() const
{
  bool valid = fullyInitialized; //can only be valid if fully initialized

  if(dmp_endPosition.size() != dmp_startPosition.size() ||
     dmp_endPosition.size() != dmp_startVelocity.size() ||
     dmp_endPosition.size() != dmp_endVelocity.size() ||
     dmp_endPosition.size() != dmp_startAcceleration.size() ||
     dmp_endPosition.size() != dmp_endAcceleration.size())
  {
    valid = false;
    cerr << "DMPConfig not valid. All dimensions need to be equal." << endl;
  }

  if(dmp_execution_time <= 0.0)
  {
    valid = false;
    cerr << "DMPConfig not valid. Execution time should be > 0." << endl;
  }

  if(valid)
  {
    //this loop only works if all dimensions are equal
    for(unsigned i = 0; i < dmp_startPosition.size(); ++i)
    {
      if(std::isnan(dmp_startPosition[i]))
      {
        valid = false;
        cerr << "DMPConfig not valid. dmp_startPosition contains Nan." << endl;
      }
      if(std::isnan(dmp_endPosition[i]))
      {
        valid = false;
        cerr << "DMPConfig not valid. dmp_endPosition contains Nan." << endl;
      }
      if(std::isnan(dmp_startVelocity[i]))
      {
        valid = false;
        cerr << "DMPConfig not valid. dmp_startVelocity contains Nan." << endl;
      }
      if(std::isnan(dmp_endVelocity[i]))
      {
        valid = false;
        cerr << "DMPConfig not valid. dmp_endVelocity contains Nan." << endl;
      }
      if(std::isnan(dmp_startAcceleration[i]))
      {
        valid = false;
        cerr << "DMPConfig not valid. dmp_startAcceleration contains Nan." << endl;
      }
      if(std::isnan(dmp_endAcceleration[i]))
      {
        valid = false;
        cerr << "DMPConfig not valid. dmp_endAcceleration contains Nan." << endl;
      }


      if(std::isinf(dmp_startPosition[i]))
      {
        valid = false;
        cerr << "DMPConfig not valid. dmp_startPosition contains Inf." << endl;
      }
      if(std::isinf(dmp_endPosition[i]))
      {
        valid = false;
        cerr << "DMPConfig not valid. dmp_endPosition contains Inf." << endl;
      }
      if(std::isinf(dmp_startVelocity[i]))
      {
        valid = false;
        cerr << "DMPConfig not valid. dmp_startVelocity contains Inf." << endl;
      }
      if(std::isinf(dmp_endVelocity[i]))
      {
        valid = false;
        cerr << "DMPConfig not valid. dmp_endVelocity contains Inf." << endl;
      }
      if(std::isinf(dmp_startAcceleration[i]))
      {
        valid = false;
        cerr << "DMPConfig not valid. dmp_startAcceleration contains Inf." << endl;
      }
      if(std::isinf(dmp_endAcceleration[i]))
      {
        valid = false;
        cerr << "DMPConfig not valid. dmp_endAcceleration contains Inf." << endl;
      }
    }
  }
  return valid;
}





}
