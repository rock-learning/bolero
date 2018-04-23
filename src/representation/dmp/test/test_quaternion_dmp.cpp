#define PRIVATE public
#include <catch/catch.hpp>
#include <vector>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <iostream>
#include <lib_manager/LibManager.hpp>
#include "test_helpers.h"
#include "QuaternionDmpModel.h"
#include "QuaternionDmp.h"

using namespace dmp;
using namespace lib_manager;
using namespace std;
using namespace Eigen;

double centers[25] = {1, 0.829601643194907, 0.68823888639169, 0.570964111061179, 0.473672764741673, 0.392959703966367, 0.326000016119882, 0.27045014905462, 0.22436588805802, 0.186134309409818, 0.154417328941334, 0.1281048698275, 0.106276010510163, 0.0881667529514306, 0.0731432831236663, 0.0606797878680638, 0.0503400517240641, 0.0417621896288002, 0.03464598113947, 0.028742362883404, 0.0238447114773763, 0.0197816118231398, 0.0164108576735206, 0.0136144744921904, 0.0112945904099563};
double widths[25] = {79.3021357521556, 115.224724031393, 167.41966533669, 243.25807310081, 353.450056238711, 513.557230239902, 746.190371387455, 1084.20257288801, 1575.3288492202, 2288.9274064117, 3325.77459900897, 4832.29684455261, 7021.24936573615, 10201.7620691956, 14822.9957227229, 21537.573676543, 31293.7471311169, 45469.3097845492, 66066.1736551044, 95993.0846126371, 139476.403485294, 202656.964381283, 294457.300202516, 427841.707326799, 427841.707326799};

string createModelFile()
{
  const std::string yaml("---\n"
  "name: ''\n"
  "rbf_centers: [1, 0.829601643194907, 0.68823888639169, 0.570964111061179, 0.473672764741673, 0.392959703966367, 0.326000016119882, 0.27045014905462, 0.22436588805802, 0.186134309409818, 0.154417328941334, 0.1281048698275, 0.106276010510163, 0.0881667529514306, 0.0731432831236663, 0.0606797878680638, 0.0503400517240641, 0.0417621896288002, 0.03464598113947, 0.028742362883404, 0.0238447114773763, 0.0197816118231398, 0.0164108576735206, 0.0136144744921904, 0.0112945904099563]\n"
  "rbf_widths: [79.3021357521556, 115.224724031393, 167.41966533669, 243.25807310081, 353.450056238711, 513.557230239902, 746.190371387455, 1084.20257288801, 1575.3288492202, 2288.9274064117, 3325.77459900897, 4832.29684455261, 7021.24936573615, 10201.7620691956, 14822.9957227229, 21537.573676543, 31293.7471311169, 45469.3097845492, 66066.1736551044, 95993.0846126371, 139476.403485294, 202656.964381283, 294457.300202516, 427841.707326799, 427841.707326799]\n"
  "ts_alpha_z: 25\n"
  "ts_beta_z: 6.25\n"
  "ts_tau: 2\n"
  "ts_dt: 0.105263157894737\n"
  "cs_execution_time: 2\n"
  "cs_alpha: 4.08956056332224\n"
  "cs_dt: 0.105263157894737\n"
  "ft_weights: [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]\n"
  "...\n");

  char cp[9000];
  getcwd(cp, sizeof(cp));
  std::string current_path = cp;
  std::string filepath = current_path+"/quaternion_model.yml";
  TestHelpers::delete_file_if_exists(filepath);
  ofstream fout(filepath.c_str());
  assert(fout.is_open());
  fout << yaml;
  fout.close();
  return filepath;
}

string createInvalidModelFile()
{
  const std::string yaml("---\n"
                         "name: ''\n"
                         "rbf_centers: [1, 0.829601643194907, 0.68823888639169, 0.570964111061179, 0.473672764741673, 0.392959703966367, 0.326000016119882, 0.27045014905462, 0.22436588805802, 0.186134309409818, 0.154417328941334, 0.1281048698275, 0.106276010510163, 0.0881667529514306, 0.0731432831236663, 0.0606797878680638, 0.0503400517240641, 0.0417621896288002, 0.03464598113947, 0.028742362883404, 0.0238447114773763, 0.0197816118231398, 0.0164108576735206, 0.0136144744921904, 0.0112945904099563]\n"
                         "rbf_widths: [79.3021357521556, 115.224724031393, 167.41966533669, 243.25807310081, 353.450056238711, 513.557230239902, 746.190371387455, 1084.20257288801, 1575.3288492202, 2288.9274064117, 3325.77459900897, 4832.29684455261, 7021.24936573615, 10201.7620691956, 14822.9957227229, 21537.573676543, 31293.7471311169, 45469.3097845492, 66066.1736551044, 95993.0846126371, 139476.403485294, 202656.964381283, 294457.300202516, 427841.707326799, 427841.707326799]\n"
                         "ts_dt: 0.105263157894737\n"
                         "cs_execution_time: 2\n"
                         "ft_weights: [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]\n"
                         "...\n");

  char cp[9000];
  getcwd(cp, sizeof(cp));
  std::string current_path = cp;
  std::string filepath = current_path+"/quaternion_model_broken.yml";
  TestHelpers::delete_file_if_exists(filepath);
  ofstream fout(filepath.c_str());
  assert(fout.is_open());
  fout << yaml;
  fout.close();
  return filepath;
}

string createConfigFile()
{
  const std::string yaml("---\n"
                         "name: ''\n"
                         "executionTime: 2\n"
                         "startPosition: [0, 1, 2, 3]\n"
                         "endPosition: [4, 5, 6, 7]\n"
                         "startVelocity: [8, 9, 10]\n"
                         "...\n");

  char cp[9000];
  getcwd(cp, sizeof(cp));
  std::string current_path = cp;
  std::string filepath = current_path+"/quaternion_config.yml";
  TestHelpers::delete_file_if_exists(filepath);
  ofstream fout(filepath.c_str());
  assert(fout.is_open());
  fout << yaml;
  fout.close();
  return filepath;
}

TEST_CASE("initialize without model", "[QuaternionDmp]")
{
  LibManager manager;
  QuaternionDmp dmp(&manager);
  QuaternionDmpModel model;
  const bool ret = dmp.initialize("some/file/that/does/not/exist.yaml");
  REQUIRE(ret == false);
}

TEST_CASE("initialize with invalid model", "[QuaternionDmp]")
{
  LibManager manager;
  QuaternionDmp dmp(&manager);
  QuaternionDmpModel model;
  try
  {//the yaml parsers throws exceptions...
    const bool ret = dmp.initialize(createInvalidModelFile());
    REQUIRE(false);
  }
  catch(...)
  {
    REQUIRE(true);
  }
}

TEST_CASE("configure with empty string" ,"[QuaternionDmp]")
{
  LibManager manager;
  QuaternionDmp dmp(&manager);
  REQUIRE(false == dmp.configureYaml(" "));
}

TEST_CASE("configure" ,"[QuaternionDmp]")
{
  LibManager manager;
  QuaternionDmp dmp(&manager);
  REQUIRE(dmp.initialize(createModelFile()));
  REQUIRE(dmp.configure(createConfigFile()));

  Quaterniond start(0, 1, 2, 3);
  start.normalize();
  REQUIRE(dmp.startData(0) == start.w());
  REQUIRE(dmp.startData(1) == start.vec().x());
  REQUIRE(dmp.startData(2) == start.vec().y());
  REQUIRE(dmp.startData(3) == start.vec().z());

  Quaterniond end(4, 5, 6, 7);
  end.normalize();

  REQUIRE(dmp.goalData(0) == end.w());
  REQUIRE(dmp.goalData(1) == end.vec().x());
  REQUIRE(dmp.goalData(2) == end.vec().y());
  REQUIRE(dmp.goalData(3) == end.vec().z());
  REQUIRE(dmp.startData(4) == 8);
  REQUIRE(dmp.startData(5) == 9);
  REQUIRE(dmp.startData(6) == 10);

  REQUIRE(dmp.initialized);
}


TEST_CASE("stepping", "[QuaternionDmp]")
{
  LibManager manager;
  QuaternionDmp dmp(&manager);
  REQUIRE(dmp.initialize(createModelFile()));
  REQUIRE(dmp.configure(createConfigFile()));

  double pos[4] = {0, 1, 2, 3};
  while(dmp.canStep())
  {
    dmp.setInputs(&pos[0], 4);
    Quaterniond temp(pos[0], pos[1], pos[2], pos[3]);
    temp.normalize();
    REQUIRE(dmp.lastData(0) == temp.w());
    REQUIRE(dmp.lastData(1) == temp.vec().x());
    REQUIRE(dmp.lastData(2) == temp.vec().y());
    REQUIRE(dmp.lastData(3) == temp.vec().z());
    dmp.step();
    dmp.getOutputs(&pos[0], 4);
  }

  Quaterniond end(4, 5, 6, 7);
  end.normalize();
  REQUIRE(pos[0] == Approx(end.w()).epsilon(0.1));
  REQUIRE(pos[1] == Approx(end.vec().x()).epsilon(0.1));
  REQUIRE(pos[1] == Approx(end.vec().y()).epsilon(0.1));
  REQUIRE(pos[2] == Approx(end.vec().z()).epsilon(0.1));
}


