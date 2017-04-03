#pragma once
#include <string>
#include <vector>

namespace YAML
{
class Parser;
}

namespace dmp
{
struct QuaternionDmpConfig
{
  std::string config_name;
  double executionTime; /**<Execution time of the dmp. I.e. time that it takes to get from the starting position to the end position */
  std::vector<double> startPosition; /**<Start rotation (w, x, y, z)*/
  std::vector<double> endPosition; /**<End rotation (w, x, y, z) */
  std::vector<double> startVelocity; /**<Start velocity as angle axis (x, y, z) */
  bool fullyInitialized;/**<True if all attributes have been initialized from yaml */

  QuaternionDmpConfig();
  bool fromYamlFile(const std::string& filepath, const std::string& name);
  bool fromYamlString(const std::string& yaml, const std::string& name);
  bool fromIstream(std::istream& stream, std::string name);
  void toYamlFile(std::string filepath);
  bool isValid() const;

};
}
