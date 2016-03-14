#include <catch/catch.hpp>
#include "DMPModel.h"
#include "test_helpers.h"
#include <fstream>
#include <string>
using namespace dmp_cpp;
using namespace std;


bool have_model=false;
DMPModel initial_model;
bool have_other_model=false;
DMPModel other_model;

void make_model(){
    if(have_model)
        return;
    initial_model.model_name = "a name";
    initial_model.rbf_centers = TestHelpers::random_vector(5);
    initial_model.rbf_widths = TestHelpers::random_vector(5);
    initial_model.ts_alpha_z = TestHelpers::random_double();
    initial_model.ts_beta_z = TestHelpers::random_double();
    initial_model.ts_tau = TestHelpers::random_double();
    initial_model.cs_execution_time = initial_model.ts_tau; //needs to be divisible by dt
    initial_model.ts_dt = initial_model.ts_tau / TestHelpers::random_int();
    initial_model.cs_alpha = TestHelpers::random_double();
    initial_model.cs_dt = initial_model.ts_dt; //there can only be one dt
    //initial_model.fop_coefficients = random_vector(5);
    initial_model.ft_weights = TestHelpers::random_matrix(5,8);

    //std::cout << "Initial model:\n"<<initial_model<<std::endl;
    have_model = true;
}

void make_other_model(){
    if(have_other_model)
        return;
    other_model.model_name = "another name";
    other_model.rbf_centers = TestHelpers::random_vector(5);
    other_model.rbf_widths = TestHelpers::random_vector(5);
    other_model.ts_alpha_z = TestHelpers::random_double();
    other_model.ts_beta_z = TestHelpers::random_double();
    other_model.ts_tau = TestHelpers::random_double();
    other_model.cs_execution_time = other_model.ts_tau; //needs to be divisible by dt
    other_model.ts_dt = other_model.ts_tau / TestHelpers::random_int();
    other_model.cs_alpha = TestHelpers::random_double();
    other_model.cs_dt = other_model.ts_dt; //there can only be one dt

    //other_model.fop_coefficients = random_vector(5);
    other_model.ft_weights = TestHelpers::random_matrix(5,8);

    //std::cout << "Initial other model:\n"<<initial_model<<std::endl;
    have_other_model = true;
}

void compare_models(DMPModel model_a, DMPModel model_b){
    REQUIRE(model_a.cs_execution_time == Approx(model_b.cs_execution_time));
    REQUIRE(model_a.cs_alpha == Approx(model_b.cs_alpha));
    REQUIRE(model_a.cs_dt == Approx(model_b.cs_dt));
    //for(uint i=0; i<loaded_model.fop_coefficients.size(); i++)
    //    REQUIRE(loaded_model.fop_coefficients[i] == Approx(initial_model.fop_coefficients[i]));
    for(uint i=0; i<model_a.rbf_centers.size(); i++)
        REQUIRE(model_a.rbf_centers[i] == Approx(model_b.rbf_centers[i]));
    for(uint i=0; i<model_a.rbf_widths.size(); i++)
        REQUIRE(model_a.rbf_widths[i] == Approx(model_b.rbf_widths[i]));
    REQUIRE(model_a.ts_alpha_z == Approx(model_b.ts_alpha_z));
    REQUIRE(model_a.ts_beta_z == Approx(model_b.ts_beta_z));
    REQUIRE(model_a.ts_dt == Approx(model_b.ts_dt));
    REQUIRE(model_a.ts_tau == Approx(model_b.ts_tau));
    REQUIRE(model_a.model_name == model_b.model_name);
    for(uint i=0; i<model_a.ft_weights.size(); i++){
        for(uint j=0; j<model_a.ft_weights[i].size(); j++){
            REQUIRE(model_a.ft_weights[i][j] == Approx(model_b.ft_weights[i][j]));
        }
    }
}



TEST_CASE("Test DMPModel", "[DMPModel]") {
    make_model();
    make_other_model();

    char cp[1000];
    get_current_dir(cp, sizeof(cp));
    std::string current_path = cp;
    std::string filepath = current_path+"/import_export_test.yml";

    SECTION("Could Export to yaml") {
        //First check if test file is already there, delete if so
        TestHelpers::delete_file_if_exists(filepath);

        //Export
        REQUIRE_NOTHROW(initial_model.to_yaml_file(filepath));

        //Does file exist?
        std::ifstream ifile;
        ifile.open(filepath.c_str());
        REQUIRE(ifile.is_open());
        ifile.close();
    }

    SECTION("Could import") {
        DMPModel loaded_model;
        REQUIRE(loaded_model.from_yaml_file(filepath, initial_model.model_name));

        compare_models(loaded_model, initial_model);
    }

    SECTION("Could append to file"){
        DMPModel loaded_model;

        REQUIRE_NOTHROW(other_model.to_yaml_file(filepath));
        REQUIRE(loaded_model.from_yaml_file(filepath, initial_model.model_name));
        REQUIRE(loaded_model.model_name == initial_model.model_name);

        REQUIRE(loaded_model.from_yaml_file(filepath, other_model.model_name));
        REQUIRE(loaded_model.model_name == other_model.model_name);
    }
}

TEST_CASE("from_yaml_string", "[DMPModel]")
{
    string yaml = "---\n"
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
                  "...\n";

    DMPModel model;
    model.from_yaml_string(yaml, "");
    REQUIRE(model.ts_dt == Approx(0.105263157894737));
    REQUIRE(model.ts_beta_z == Approx(6.25));
    REQUIRE(model.ts_tau == Approx(2));
    REQUIRE(model.ts_alpha_z == Approx(25));
    REQUIRE(model.cs_alpha == Approx(4.08956056332224));
    REQUIRE(model.cs_execution_time == Approx(2));
    REQUIRE(model.cs_dt == Approx(0.105263157894737));
}



