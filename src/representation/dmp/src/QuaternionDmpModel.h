#pragma once
#include "DMPModel.h"

namespace dmp
{
/**
* The DMPModel can be used for quaternion dmps as well
*/
class QuaternionDmpModel : public dmp_cpp::DMPModel{
public:
  QuaternionDmpModel(const std::string& yamlFile, const std::string& name) :
          DMPModel(yamlFile, name)
  {}
  QuaternionDmpModel() : DMPModel() {}

  QuaternionDmpModel(const DMPModel& other) : DMPModel(other)
  {  }
};
}
