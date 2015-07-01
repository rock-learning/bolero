#include "catch.hpp"
#include "DMPConfig.h"
#include "test_helpers.h"
#include <fstream>
#include <string>
#include <stdio.h>  /* defines FILENAME_MAX */
#include <unistd.h>

using namespace dmp_cpp;

bool have_config=false;
DMPConfig initial_config;
bool have_other_config=false;
DMPConfig other_config;

void make_config(){
    if(have_config)
        return;
    initial_config.config_name = "a name";
    initial_config.dmp_execution_time = TestHelpers::random_double();
    initial_config.dmp_startPosition = TestHelpers::random_vector(5);
    initial_config.dmp_endPosition = TestHelpers::random_vector(5);
    initial_config.dmp_startVelocity = TestHelpers::random_vector(5);
    initial_config.dmp_endVelocity = TestHelpers::random_vector(5);
    initial_config.dmp_startAcceleration = TestHelpers::random_vector(5);
    initial_config.dmp_endAcceleration = TestHelpers::random_vector(5);
    have_config = true;
}

void make_other_config(){
    if(have_other_config)
        return;
    other_config.config_name = "another name";
    other_config.dmp_execution_time = TestHelpers::random_double();
    other_config.dmp_startPosition = TestHelpers::random_vector(5);
    other_config.dmp_endPosition = TestHelpers::random_vector(5);
    other_config.dmp_startVelocity = TestHelpers::random_vector(5);
    other_config.dmp_endVelocity = TestHelpers::random_vector(5);
    other_config.dmp_startAcceleration = TestHelpers::random_vector(5);
    other_config.dmp_endAcceleration = TestHelpers::random_vector(5);
    have_other_config = true;
}

void compare_configs(DMPConfig config_a, DMPConfig config_b)
{

  REQUIRE(config_a.dmp_startPosition.size() == config_a.dmp_endPosition.size());
  REQUIRE(config_a.dmp_startPosition.size() == config_a.dmp_startVelocity.size());
  REQUIRE(config_a.dmp_startPosition.size() == config_a.dmp_endVelocity.size());
  REQUIRE(config_a.dmp_startPosition.size() == config_a.dmp_startAcceleration.size());
  REQUIRE(config_a.dmp_startPosition.size() == config_a.dmp_endAcceleration.size());

  REQUIRE(config_a.dmp_startPosition.size() == config_b.dmp_startPosition.size());
  REQUIRE(config_a.dmp_endPosition.size() == config_b.dmp_endPosition.size());
  REQUIRE(config_a.dmp_startVelocity.size() == config_b.dmp_startVelocity.size());
  REQUIRE(config_a.dmp_endVelocity.size() == config_b.dmp_endVelocity.size());
  REQUIRE(config_a.dmp_startAcceleration.size() == config_b.dmp_startAcceleration.size());
  REQUIRE(config_a.dmp_endAcceleration.size() == config_b.dmp_endAcceleration.size());

  REQUIRE(config_a.dmp_execution_time == Approx(config_b.dmp_execution_time));

  for(unsigned i = 0; i < config_a.dmp_startPosition.size(); ++i)
  {
    REQUIRE(config_a.dmp_startPosition[i] == Approx(config_b.dmp_startPosition[i]));
    REQUIRE(config_a.dmp_endPosition[i] == Approx(config_b.dmp_endPosition[i]));
    REQUIRE(config_a.dmp_startVelocity[i] == Approx(config_b.dmp_startVelocity[i]));
    REQUIRE(config_a.dmp_endVelocity[i] == Approx(config_b.dmp_endVelocity[i]));
    REQUIRE(config_a.dmp_startAcceleration[i] == Approx(config_b.dmp_startAcceleration[i]));
    REQUIRE(config_a.dmp_endAcceleration[i] == Approx(config_b.dmp_endAcceleration[i]));
  }
}



TEST_CASE("Test DMPConfig", "[DMPConfig]") {
    make_config();
    make_other_config();

    char cp[1000];
    getcwd(cp, sizeof(cp));
    std::string current_path = cp;
    std::string filepath = current_path+"/import_export_test_config.yml";

    SECTION("Could Export to yaml") {
        //First check if test file is already there, delete if so
        TestHelpers::delete_file_if_exists(filepath);

        //Export
        REQUIRE_NOTHROW(initial_config.to_yaml_file(filepath));

        //Does file exist?
        std::ifstream ifile;
        ifile.open(filepath.c_str());
        REQUIRE(ifile.is_open());
        ifile.close();
    }

    SECTION("Could import") {
        DMPConfig loaded_config;
        REQUIRE(loaded_config.from_yaml_file(filepath, initial_config.config_name));

        compare_configs(loaded_config, initial_config);
    }

    SECTION("Could append to file"){
        DMPConfig loaded_config;

        REQUIRE_NOTHROW(other_config.to_yaml_file(filepath));
        REQUIRE(loaded_config.from_yaml_file(filepath, initial_config.config_name));
        REQUIRE(loaded_config.config_name == initial_config.config_name);

        REQUIRE(loaded_config.from_yaml_file(filepath, other_config.config_name));
        REQUIRE(loaded_config.config_name == other_config.config_name);
    }

    SECTION("load from string") {
      DMPConfig config;
      const std::string yaml("name: ''\n"
                             "dmp_execution_time: 10\n"
                             "dmp_startPosition: [0, 1]\n"
                             "dmp_endPosition: [10, 11]\n"
                             "dmp_startVelocity: [1, 2]\n"
                             "dmp_endVelocity: [3, 4]\n"
                             "dmp_startAcceleration: [5, 6]\n"
                             "dmp_endAcceleration: [7, 8]");
      REQUIRE(config.from_yaml_string(yaml,""));
      REQUIRE(config.dmp_execution_time == 10.0);
      REQUIRE(config.dmp_startPosition[0] == 0.0);
      REQUIRE(config.dmp_startPosition[1] == 1.0);
      REQUIRE(config.dmp_endPosition[0] == 10.0);
      REQUIRE(config.dmp_endPosition[1] == 11.0);
      REQUIRE(config.dmp_startVelocity[0] == 1.0);
      REQUIRE(config.dmp_startVelocity[1] == 2.0);
      REQUIRE(config.dmp_endVelocity[0] == 3.0);
      REQUIRE(config.dmp_endVelocity[1] == 4.0);
      REQUIRE(config.dmp_startAcceleration[0] == 5.0);
      REQUIRE(config.dmp_startAcceleration[1] == 6.0);
      REQUIRE(config.dmp_endAcceleration[0] == 7.0);
      REQUIRE(config.dmp_endAcceleration[1] == 8.0);
    }
}


