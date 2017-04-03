#pragma once

#include <string>
#include <vector>
#include <istream>

namespace dmp_cpp{
struct DMPConfig{
    std::string config_name;
    double dmp_execution_time; /**<Execution time of the dmp. I.e. time that it takes to get from the starting position to the end position */
    std::vector<double> dmp_startPosition; /**<Start position of the trajectory */
    std::vector<double> dmp_endPosition; /**<End position of the trajectory */
    std::vector<double> dmp_startVelocity; /**<Start velocity of the trajectory */
    std::vector<double> dmp_endVelocity; /**<End velocity of the trajectory */
    std::vector<double> dmp_startAcceleration; /**<Start Acceleration of the trajectory */
    std::vector<double> dmp_endAcceleration; /**<End Acceleration of the trajectory */

    DMPConfig();
    DMPConfig(const std::string& filepath, const std::string& name);

    bool from_yaml_file(const std::string& filepath, const std::string& name);
    bool from_yaml_string(const std::string& yaml, const std::string& name);
    bool from_yaml_istream(std::istream& stream, std::string name);
    void to_yaml_file(std::string filepath);

    bool is_valid() const;

    //True if all attributes have been initialized by any of the from_yaml methods
    bool fullyInitialized;
};
}

