#pragma once
#include <string>
#include "QuaternionDmpConfig.h"
#include "DMPConfig.h"

namespace YAML
{
class Parser;
}

namespace dmp
{

/**
* A rigid body dmp is a simple aggregation of a QuaternionDmp and
* a normal dmp. Therefore the config is an aggregation of the corresponding
* configs.
*/
class RigidBodyDmpConfig
{
public:
  QuaternionDmpConfig rotationConfig;
  dmp_cpp::DMPConfig translationConfig;
  bool fromYamlFile(const std::string& filepath, const std::string& name);
  bool fromYamlString(const std::string& yaml, const std::string& name);
  bool fromYamlParser(YAML::Parser& parser,std::string name);
  void toYamlFile(std::string filepath);
  bool isValid() const;

private:
  bool fullyInitialized;
};
}
