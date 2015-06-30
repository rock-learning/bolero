/**
 * @file DmpBehavior.h
 *  @author Arne Boeckmann (arne.boeckmann@dfki.de)
 */

#pragma once
#include <Behavior.h>
#include <LoadableBehavior.h>
#include <string>
#include "DMPModel.h"
#include "DMPConfig.h"
#include <Eigen/Core>
#include <lib_manager/LibInterface.hpp>

namespace dmp {
class Dmp;

/**
 * A simple wrapper around the dmp class that provides the Behavior interface.
 *
 * \note This class expects (and enforces) the following call order:
 *       Initialization:
 *       1. initialize()
 *       2. configure()
 *
 *       loop:
 *       3. setInputs()
 *       4. step()
 *       5. getOutputs()
 *
 *       configure() can be called at any time during the loop to
 *       re-configure the dmp.
 */
class DmpBehavior : public bolero::LoadableBehavior
{
public:

  DmpBehavior(lib_manager::LibManager *manager);

  /**
   * \param values Concatenation of position, velocity and acceleration values
   */
  virtual void setInputs(const double *values, int numInputs);

  /**
   * \param values Concatenation of position, velocity and acceleration values
   */
  virtual void getOutputs(double *values, int numOutputs) const;

  virtual void step();

  virtual bool canStep() const;

  /**
   * \param modelPath Path to a yaml file that can be interpreted as dmp_cpp::DMPModel
   */
  virtual bool initialize(const std::string& modelPath);

  /**Initialize the dmp using the specified model */
  virtual bool initialize(const dmp_cpp::DMPModel& model);

  /**
   * \param configPath Path to a yaml file that can be interpreted as dmp_cpp::DMPConfig
   */
  virtual bool configure(const std::string& configPath);
  /**
   * \param yaml A string containing yaml that can be interpreted as dmp_cpp::DMPConfig
   */
  virtual bool configureYaml(const std::string& yaml);

  bool configure(const dmp_cpp::DMPConfig& config);

  /**Returns the underlying dmp*/
  Dmp& getDmp();

  CREATE_MODULE_INFO();

private:
  dmp_cpp::DMPModel model;
  dmp_cpp::DMPConfig config;
  std::auto_ptr<Dmp> dmp;
  Eigen::ArrayXd data;//contains concatenation of position, velocity and acceleration

  //This class needs a certain call order to function properly.
  //This enum is used to enforce the call order
  enum State
  {
    INITIALIZE,
    CONFIGURE,
    SET_INPUTS,
    STEP,
    GET_OUTPUTS
  };
  mutable State expectedState;//contains the function that should be called next
  bool stepPossible; //True if step() can be called at least one more time
};

}
