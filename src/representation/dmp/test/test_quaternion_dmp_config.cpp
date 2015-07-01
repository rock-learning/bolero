#include "catch.hpp"
#include "QuaternionDmpConfig.h"
#include "test_helpers.h"
#include <fstream>
#include <string>
#include <stdio.h>  /* defines FILENAME_MAX */
#include <unistd.h>

using namespace dmp;


TEST_CASE("to/from yaml file", "[QuaternionDmpConfig]")
{
  QuaternionDmpConfig cfg;
  cfg.startPosition.push_back(0);
  cfg.startPosition.push_back(1);
  cfg.startPosition.push_back(2);
  cfg.startPosition.push_back(3);
  cfg.endPosition.push_back(4);
  cfg.endPosition.push_back(5);
  cfg.endPosition.push_back(6);
  cfg.endPosition.push_back(7);
  cfg.startVelocity.push_back(8);
  cfg.startVelocity.push_back(9);
  cfg.startVelocity.push_back(10);


  char cp[9000];
  getcwd(cp, sizeof(cp));
  std::string current_path = cp;
  std::string filepath = current_path+"/quaternion_import_export_test.yml";
  TestHelpers::delete_file_if_exists(filepath);
  cfg.toYamlFile(filepath);
  QuaternionDmpConfig cfg2;
  REQUIRE(cfg2.fromYamlFile(filepath, ""));
  REQUIRE(cfg2.startPosition[0] == Approx(cfg.startPosition[0]));
  REQUIRE(cfg2.startPosition[1] == Approx(cfg.startPosition[1]));
  REQUIRE(cfg2.startPosition[2] == Approx(cfg.startPosition[2]));
  REQUIRE(cfg2.startPosition[3] == Approx(cfg.startPosition[3]));
  REQUIRE(cfg2.startPosition.size() == 4);
  REQUIRE(cfg2.endPosition[0] == Approx(cfg.endPosition[0]));
  REQUIRE(cfg2.endPosition[1] == Approx(cfg.endPosition[1]));
  REQUIRE(cfg2.endPosition[2] == Approx(cfg.endPosition[2]));
  REQUIRE(cfg2.endPosition[3] == Approx(cfg.endPosition[3]));
  REQUIRE(cfg2.endPosition.size() == 4);
  REQUIRE(cfg2.startVelocity[0] == Approx(cfg.startVelocity[0]));
  REQUIRE(cfg2.startVelocity[1] == Approx(cfg.startVelocity[1]));
  REQUIRE(cfg2.startVelocity[2] == Approx(cfg.startVelocity[2]));
  REQUIRE(cfg2.startVelocity.size() == 3);
}


TEST_CASE("from yaml string", "[QuaternionDmpConfig]")
{
  const std::string yaml("name: 'aa'\n"
          "startPosition: [0, 1, 2, 3]\n"
          "endPosition: [4, 5, 6, 7]\n"
          "startVelocity: [8, 9, 10]\n");

  QuaternionDmpConfig cfg;
  cfg.fromYamlString(yaml, "aa");
  REQUIRE(cfg.startPosition[0] == Approx(0));
  REQUIRE(cfg.startPosition[1] == Approx(1));
  REQUIRE(cfg.startPosition[2] == Approx(2));
  REQUIRE(cfg.startPosition[3] == Approx(3));
  REQUIRE(cfg.startPosition.size() == 4);
  REQUIRE(cfg.endPosition[0] == Approx(4));
  REQUIRE(cfg.endPosition[1] == Approx(5));
  REQUIRE(cfg.endPosition[2] == Approx(6));
  REQUIRE(cfg.endPosition[3] == Approx(7));
  REQUIRE(cfg.endPosition.size() == 4);
  REQUIRE(cfg.startVelocity[0] == Approx(8));
  REQUIRE(cfg.startVelocity[1] == Approx(9));
  REQUIRE(cfg.startVelocity[2] == Approx(10));
  REQUIRE(cfg.startVelocity.size() == 3);
  REQUIRE(cfg.config_name == "aa");

}