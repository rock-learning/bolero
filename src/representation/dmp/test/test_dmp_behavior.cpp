#include <catch/catch.hpp>
#include "DmpBehavior.h"
#include "DMPModel.h"
#include "test_helpers.h"
#include <stdio.h>  /* defines FILENAME_MAX */
#include <unistd.h>
#include <iostream>
#include <lib_manager/LibManager.hpp>
#include <yaml-cpp/yaml.h>
#include <vector>
#include <sstream>

#define get_current_dir getcwd

using namespace dmp;
using namespace dmp_cpp;
using namespace std;
using namespace Eigen;

const int numPhases = 20;
const double numCenters = 25;
const double lastPhaseValue = 0.01;
double executionTime = 2;
const double overlap = 0.1;
const double dt = executionTime/(numPhases - 1);
const double alpha = 5.0;

ArrayXXd weights(2,2);
ArrayXd startPos(2);
ArrayXd startVel(2);
ArrayXd startAcc(2);
ArrayXd endPos(2);
ArrayXd endVel(2);
ArrayXd endAcc(2);

void initArrays()
{
  startPos << 0, 0;
  startVel << 0, 0;
  startAcc << 0, 0;
  endPos << 10, 10;
  endVel << 0, 0;
  endAcc << 0, 0;
  weights = ArrayXXd::Random(2, numCenters) * 20;
}


std::string getFilePath(const std::string& name)
{
  char cp[1000];
  get_current_dir(cp, sizeof(cp));
  std::string current_path = cp;
  std::string filepath = current_path + "/" + name;
  return filepath;
}

TEST_CASE("Reconfigure midrun", "[DmpBehavior]")
{
  lib_manager::LibManager manager;
  DmpBehavior behav(&manager);
  behav.initialize(getFilePath("model.yaml"));
  behav.configure(getFilePath("config.yaml"));

  double data[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  //run half of the steps with default config
  for(int i = 0; i < numPhases/2; ++i)
  {
    behav.setInputs(&data[0], 6);
    behav.step();
    behav.getOutputs(&data[0], 6);
  }

  //change goal using the config yaml
  vector<double> newPos, newVel, newAcc;
  newPos.push_back(5);
  newPos.push_back(5);
  newVel.push_back(1);
  newVel.push_back(1);
  newAcc.push_back(0);
  newAcc.push_back(0);
  YAML::Emitter out;
  out << YAML::BeginDoc;
  out << YAML::BeginMap << YAML::Key << "name" << YAML::Value << "a";
  out << YAML::Key << "dmp_endPosition" << YAML::Value << YAML::Flow << newPos;
  out << YAML::Key << "dmp_endVelocity" << YAML::Value << YAML::Flow << newVel;
  out << YAML::Key << "dmp_endAcceleration" << YAML::Value << YAML::Flow << newAcc;
  out << YAML::EndMap;
  out << YAML::EndDoc;

  REQUIRE(behav.configureYaml(out.c_str()));

  for(int i = numPhases/2; i < numPhases; ++i)
  {
    behav.setInputs(&data[0], 6);
    behav.step();
    behav.getOutputs(&data[0], 6);
  }
  //we only care about the position because the dmp will not reach the
  //desired velocity and acceleration in this case.
  //even the position is off by ~0.3
  REQUIRE(data[0] == Approx(newPos[0]).epsilon(0.3));
  REQUIRE(data[1] == Approx(newPos[1]).epsilon(0.3));

}



