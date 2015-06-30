#pragma once

#include "Dmp.h"
#include "DMPModel.h"
#include "DMPConfig.h"
#include <memory>
#include <cassert>
#include <string>
namespace dmp_cpp{

class DMPWrapper{
protected:
    std::string dmp_name_;
    std::auto_ptr<dmp::Dmp> dmp_;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    inline DMPWrapper() {}
    inline DMPWrapper(std::string filepath, std::string name){
        init_from_yaml(filepath, name);
    }

    inline void init_from_model(const dmp_cpp::DMPModel& model){
        dmp_.reset(new dmp::Dmp(model));
        dmp_name_ = model.model_name;
    }

    inline dmp_cpp::DMPModel generate_model(){
        return dmp_->generateModel();
    }

    inline void init_from_dmp(const dmp::Dmp& dmp){
        dmp_.reset(new dmp::Dmp(dmp));
    }

    inline void init_from_yaml(std::string filepath, std::string name){
        DMPModel model;
        model.from_yaml_file(filepath, name);
        init_from_model(model);
    }

    inline void apply_config(const DMPConfig& cfg){
      assert(dmp_.get() != NULL && "Tried to access dmp before initializing");
      dmp_->initialize(cfg);
    }

    inline dmp::Dmp& dmp(){
        assert(dmp_.get() != NULL && "Tried to access dmp before initializing");
        return *(dmp_.get());
    }

};
}
