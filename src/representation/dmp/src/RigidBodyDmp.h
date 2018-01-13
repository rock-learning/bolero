#pragma once
#include "QuaternionDmp.h"
#include "DmpBehavior.h"
#include "RigidBodyDmpConfig.h"
#include <LoadableBehavior.h>
#include <lib_manager/LibInterface.hpp>
#include <string>
#include <memory>
#include <Eigen/Core>

namespace dmp
{

/**
* A behavior that can be used to specify trajectories of 3d poses.
* A pose is the translation and orientation of an object in R^3.
* The translation is defined by a 3d vector.
* The rotation is defined by a quaternion.
*/
class RigidBodyDmp : public bolero::LoadableBehavior
{
public:

  RigidBodyDmp(lib_manager::LibManager *manager);

  /**
   * @param values Format: [p_x, p_y, p_z, v_x, v_y, v_z, a_x, a_y, a_z, w, x, y, z]
   *               p = position, v = velocity, a = acceleration
   *               [w, x, y, z] = rotation as quaternion
   * @param numInputs number of inputs, must be 13
   */
  virtual void setInputs(const double *values, int numInputs);

  /**
  * @param[out] values Format: [p_x, p_y, p_z, v_x, v_y, v_z, a_x, a_y, a_z, w, x, y, z]
  *                            p = position, v = velocity, a = acceleration
  *                            [w, x, y, z] = rotation as quaternion
   * @param numOutputs number of outputs, must be 13
  */
  virtual void getOutputs(double *values, int numOutputs) const;

  /** Compute output for the received input. */
  virtual void step();
  /** Check if more steps are possible. */
  virtual bool canStep() const;

  /** Initialize from yaml file
   * @param modelPath path to the model definition
   */
  virtual bool initialize(const std::string& modelPath);
  /** Initialize from yaml string
   * @param yaml model definition in YAML format
   */
  virtual bool initializeYaml(const std::string yaml);
  /** Initialize from DMPModel
   * @param model model definition
   */
  virtual bool initialize(const dmp_cpp::DMPModel& model);

 //FIXME right now reconfiguration is not possible. configure() can only be called once
  /** Configure from yaml file
   * @param configPath path to the configuration file
   */
  virtual bool configure(const std::string& configPath);
  /** Configure from yaml string
   * @param yaml configuration in YAML format
   */
  virtual bool configureYaml(const std::string& yaml);
  /**
   * Configure from RigidBodyDmpConfig
   * @param config configuration
   */
  virtual bool configure(const RigidBodyDmpConfig& config);

  /** Sets the weights.
   * @param weights new weight array
   * @param rows should always be 6
   * @param cols should always be the same as the columns in centers and widths
   */
  void setWeights(const double* weights, const int rows, const int cols);

  /** Gets the weights.
   * @param[out] weights current weight array, will be filled
   * @param rows should always be 6
   * @param cols should always be the same as the columns in centers and widths
   */
  void getWeights(double* weights, const int rows, const int cols);


  //creates the module info needed by the lib manager.
  //without it the lib manager would be unable to load this module at run time.
  CREATE_MODULE_INFO();

private:
  lib_manager::LibManager* manager;
  RigidBodyDmpConfig config;
  std::auto_ptr<DmpBehavior> translationDmp;
  std::auto_ptr<QuaternionDmp> rotationDmp;
  bool initialized; /**<True if initialize() has been called successfully */
  bool configured; /**<True if configure() has been called successfully */
};

}
