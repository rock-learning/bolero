#pragma once
#include <Eigen/Geometry>
#include <LoadableBehavior.h>
#include "QuaternionDmpModel.h"
#include "QuaternionDmpConfig.h"


namespace dmp
{

class CanonicalSystem;
class RbfFunctionApproximator;
class ForcingTerm;

/**
 * A quaternion dmp describes three dimensional rotation trajectories.
 * Example usage:
 * @code
 *   ...
 *   QuaternionDmp dmp(manager);
 *   dmp.initialize("/path/to/init.yaml");
 *   dmp.configure("/path/to/config.yaml");
 *
 *   double position[4] = {initial position};
 *   while(dmp.canStep())
 *   {
 *     dmp.setInputs(&position[0], 4);
 *     dmp.step();
 *     dmp.getOutputs(&position[0], 4);
 *     //do something with the output
 *   }
 * @endcode
 *
 * This implementation is based on [Ude2014]
*/
class QuaternionDmp :  public bolero::LoadableBehavior
{
/*note: the QuaternionDmp does not extend Dmp because dmp contains template methods that cannot be overridden.
        CRTP would be a solution to this problem but it is complex to implement.
 */

public:
  QuaternionDmp(lib_manager::LibManager *manager);

  /** Initialize from yaml file
   * @param initialConfigPath path to the model definition
   */
  virtual bool initialize(const std::string& initialConfigPath);

  /** Initialize from QuaternionDmpModel
   * @param model model definition
   */
  virtual bool initialize(const QuaternionDmpModel& model);

  /** Configure from yaml file
   * @param configPath path to the configuration file
   * @see configure(QuaternionDmpConfig)
   */
  virtual bool configure(const std::string& configPath);

  /** Configure from yaml string
   * @param yaml configuration in YAML format
   * @see configure(QuaternionDmpConfig)*/
   */
  virtual bool configureYaml(const std::string& yaml);

  /**Applies the given configuration.
   *  This can be done at any time after initialize() has been called.
   *  The dmp can be reconfigured mid-run, e.g. to change the goal.
   *
   * @param config configuration
   *
   *  @note The start and end positions will be normalized. Therefore the values
   *        might differ from the values defined in the configuration file.
   */
  bool configure(const QuaternionDmpConfig& config);

  /**
  * @param[in] values array of size >= 4 that represents a quaternion. [w, x, ,y, z]
  *                   Only the first 4 elements of the array will be used.
  * @param numInputs number of inputs, must be 4
  *
  * @note The inputs will be normalized
  */
  virtual void setInputs(const double *values, int numInputs);

  /**
   * @param[out] values array of size >= 4 that represents a quaternion. [w, x, ,y, z]
   *                    The first 4 elements will be overwritten with the output.
   * @param numOutputs number of outputs, must be 4
   */
  virtual void getOutputs(double *values, int numOutputs) const;

  /** Compute output for the received input. */
  virtual void step();
  /** Check if more steps are possible. */
  virtual bool canStep() const;

  /**
  * Sets the weights matrix of the forcing term.
  *
  * @param newWeights A 3xN matrix. N is the number of centers in the function
  *                   approximator.
  */
  virtual void setWeights(const Eigen::ArrayXXd& newWeights);

  /**
  * Gets the weights matrix of the forcing term
  * @return A 3xN matrix. N is the number of centers in the function approximator.
  */
  virtual const Eigen::MatrixXd getWeights();

  //creates the module info needed by the lib manager.
  //without it the lib manager would be unable to load this module at run time.
  CREATE_MODULE_INFO();

private:
  QuaternionDmpConfig config;
  bool initialized;/**<If true initialize() has been called successfully */

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
