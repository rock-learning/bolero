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

/**
 * A simple wrapper around the DMP class that provides the Behavior interface.
 *
 * \note
 * This class expects (and enforces) the following call order:
 *
 * \par
 * Initialization:
 * 1. initialize()
 * 2. configure()
 *
 * \par
 * loop:
 * 3. setInputs()
 * 4. step()
 * 5. getOutputs()
 *
 * \par
 * configure() can be called at any time during the loop to re-configure the
 * dmp.
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


  /**
  * Sets the weights matrix of the forcing term
  * Should be a DxN matrix.
  * N should be equal to the number of centers in the function approximator
  * and D is the number of dimensions.
  */
  virtual void setWeights(const Eigen::ArrayXXd& newWeights);


  /**
  * Gets the weights matrix of the forcing term
  * Should be a DxN matrix.
  * N should be equal to the number of centers in the function approximator
  * and D is the number of dimensions.
  */
  virtual const Eigen::MatrixXd getWeights();

  CREATE_MODULE_INFO();

private:
  dmp_cpp::DMPModel model;
  dmp_cpp::DMPConfig config;

  //This class needs a certain call order to function properly.
  //This enum is used to enforce the call order
  enum State
  {
    INITIALIZE,
    CONFIGURE,
    CONFIGURED,
    SET_INPUTS,
    STEP,
    GET_OUTPUTS
  };
  mutable State expectedState;//contains the function that should be called next

  int taskDimensions;
  double dt;

  std::string name;
  double alphaY;
  double betaY;
  double alphaZ;
  Eigen::ArrayXd widths;
  Eigen::ArrayXd centers;
  Eigen::ArrayXXd weights;
  double startT;
  double goalT;
  Eigen::ArrayXd startData; //contains concatenation of position, velocity and acceleration
  Eigen::ArrayXd goalData; //contains concatenation of position, velocity and acceleration
  double lastT;
  double t;
  Eigen::ArrayXd lastData; //contains concatenation of position, velocity and acceleration
  Eigen::ArrayXd data; //contains concatenation of position, velocity and acceleration
};

}
