#include <catch/catch.hpp>
#include <vector>
#include <iostream>
#include <string>
#include "test_helpers.h"
#define private public
#include "RigidBodyDmpConfig.h"

using namespace dmp;
using namespace std;

void vectorCompare(const std::vector<double>& a, const std::vector<double>& b)
{
  REQUIRE(a.size() == b.size());
  for(unsigned i = 0; i < a.size(); ++i)
  {
    REQUIRE(a[i] == Approx(b[i]));
  }
}

TEST_CASE("read and write", "[RigidBodyDmpConfig]")
{
  RigidBodyDmpConfig cfg;
  cfg.rotationConfig.config_name = "test";
  cfg.translationConfig.config_name = "test";
  cfg.rotationConfig.executionTime = 42.42;
  cfg.rotationConfig.startVelocity = TestHelpers::random_vector(3);
  cfg.rotationConfig.endPosition = TestHelpers::random_vector(4);
  cfg.rotationConfig.startPosition = TestHelpers::random_vector(4);
  cfg.translationConfig.dmp_endAcceleration = TestHelpers::random_vector(3);
  cfg.translationConfig.dmp_endPosition = TestHelpers::random_vector(3);
  cfg.translationConfig.dmp_endVelocity = TestHelpers::random_vector(3);
  cfg.translationConfig.dmp_startVelocity = TestHelpers::random_vector(3);
  cfg.translationConfig.dmp_startAcceleration = TestHelpers::random_vector(3);
  cfg.translationConfig.dmp_startPosition = TestHelpers::random_vector(3);
  cfg.translationConfig.dmp_execution_time = 42.42;
  cfg.translationConfig.fullyInitialized = true;
  cfg.rotationConfig.fullyInitialized = true;
  REQUIRE(cfg.isValid());

  char cp[1000];
  getcwd(cp, sizeof(cp));
  std::string current_path = cp;
  std::string filepath = current_path+"/import_export_test_rigid_config.yml";
  TestHelpers::delete_file_if_exists(filepath);
  cfg.toYamlFile(filepath);

  RigidBodyDmpConfig cfg2;
  cfg2.fromYamlFile(filepath, "test");
  REQUIRE(cfg2.isValid());
  REQUIRE(cfg.rotationConfig.config_name == cfg2.rotationConfig.config_name);
  REQUIRE(cfg.translationConfig.config_name == cfg2.translationConfig.config_name);
  vectorCompare(cfg.rotationConfig.startVelocity, cfg2.rotationConfig.startVelocity);
  vectorCompare(cfg.rotationConfig.endPosition, cfg2.rotationConfig.endPosition);
  vectorCompare(cfg.rotationConfig.startPosition, cfg2.rotationConfig.startPosition);
  vectorCompare(cfg.translationConfig.dmp_endAcceleration, cfg2.translationConfig.dmp_endAcceleration);
  vectorCompare(cfg.translationConfig.dmp_endPosition, cfg2.translationConfig.dmp_endPosition);
  vectorCompare(cfg.translationConfig.dmp_endVelocity, cfg2.translationConfig.dmp_endVelocity);
  vectorCompare(cfg.translationConfig.dmp_startVelocity, cfg2.translationConfig.dmp_startVelocity);
  vectorCompare(cfg.translationConfig.dmp_startAcceleration, cfg.translationConfig.dmp_startAcceleration);
  vectorCompare(cfg.translationConfig.dmp_startPosition, cfg2.translationConfig.dmp_startPosition);
  REQUIRE(cfg.translationConfig.dmp_execution_time == cfg2.translationConfig.dmp_execution_time);
  REQUIRE(cfg.translationConfig.fullyInitialized == cfg2.translationConfig.fullyInitialized);
  REQUIRE(cfg.rotationConfig.fullyInitialized == cfg2.rotationConfig.fullyInitialized);


}
