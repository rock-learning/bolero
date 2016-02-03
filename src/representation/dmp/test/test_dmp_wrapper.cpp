#include "catch.hpp"
#include <fstream>
#include "DMPWrapper.h"
#include "test_helpers.h"
using namespace dmp_cpp;

#include <stdio.h>  /* defines FILENAME_MAX */
#include <unistd.h>
#define get_current_dir getcwd

extern double random_double();
extern std::vector<double> random_vector(int size);
extern std::vector<std::vector<double> > random_matrix(int size_a, int size_b);

extern bool have_model;
extern DMPModel initial_model;
extern bool have_other_model;
extern DMPModel other_model;

extern void make_model();
extern void compare_models(DMPModel model_a, DMPModel model_b);
extern void delete_file_if_exists(std::string);

TEST_CASE("Test DMPWrapper", "[DMPWrapper]") {
    make_model();

    char cp[1000];
    get_current_dir(cp, sizeof(cp));
    std::string current_path = cp;
    std::string filepath = current_path+"/import_export_test.yml";

    SECTION("DMPWrapper with DMPModel") {
        dmp_cpp::DMPWrapper wrapper;
        REQUIRE_NOTHROW(wrapper.init_from_model(initial_model));

        dmp_cpp::DMPModel other_model;
        REQUIRE_NOTHROW(other_model = wrapper.generate_model());

        compare_models(initial_model, other_model);
    }

    SECTION("DMPWrapper with DMP persitent storage") {
        //First check if test file is already there, delete if so
        TestHelpers::delete_file_if_exists(filepath);

        const int numPhases = 50;
        const int numCenters = 10;
        const double lastPhaseValue = 0.01;
        const double executionTime = 1.0;
        const double overlap = 0.1;
        const double dt = 0.01;
        const double alpha = dmp::CanonicalSystem::calculateAlpha(lastPhaseValue,
                                                                  numPhases);

        dmp::Dmp dmp(executionTime, alpha, dt, numCenters, overlap);
        double weights[numCenters] = {0.0};//note: in reality this should be a 2d array
        dmp.setWeights(&weights[0], 1, numCenters);

        dmp_cpp::DMPWrapper wrapper;
        wrapper.init_from_dmp(dmp);
        dmp_cpp::DMPModel other_model = wrapper.generate_model();
        other_model.to_yaml_file(filepath);

        dmp_cpp::DMPWrapper wrapper2;
        wrapper2.init_from_yaml(filepath,"");
        wrapper2.dmp();

        dmp_cpp::DMPWrapper wrapper3;
        dmp::Dmp dmp3 = wrapper2.dmp();
        wrapper3.init_from_dmp(dmp3);

        compare_models(wrapper3.generate_model(), wrapper.generate_model());
        compare_models(wrapper2.generate_model(), wrapper.generate_model());
    }

    SECTION("Apply config"){
        const int numPhases = 50;
        const double numCenters = 10;
        const double lastPhaseValue = 0.01;
        const double executionTime = 1.0;
        const double overlap = 0.1;
        const double dt = 0.01;
        const double alpha = dmp::CanonicalSystem::calculateAlpha(lastPhaseValue,
                                                                  numPhases);

        dmp::Dmp dmp(executionTime, alpha, dt, numCenters, overlap);

        dmp_cpp::DMPWrapper wrapper;
        wrapper.init_from_dmp(dmp);

        dmp_cpp::DMPConfig cfg;
        cfg.config_name = "a name";
        cfg.dmp_execution_time = executionTime;
        cfg.dmp_startPosition = TestHelpers::random_vector(5);
        cfg.dmp_endPosition = TestHelpers::random_vector(5);
        cfg.dmp_startVelocity = TestHelpers::random_vector(5);
        cfg.dmp_endVelocity = TestHelpers::random_vector(5);
        cfg.dmp_startAcceleration = TestHelpers::random_vector(5);
        cfg.dmp_endAcceleration = TestHelpers::random_vector(5);
        cfg.fullyInitialized = true;
        REQUIRE_NOTHROW(wrapper.apply_config(cfg));
    }
}


