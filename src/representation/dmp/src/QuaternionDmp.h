#pragma once
#include <Eigen/Geometry>
#include <LoadableBehavior.h>
#include "QuaternionDmpModel.h"
#include "QuaternionTransformationSystem.h"


namespace dmp
{

class CanonicalSystem;
class RbfFunctionApproximator;
class ForcingTerm;
class QuaternionDmpConfig;

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

  /**
  * \see QuaternionTransformationSystem::determineForces()
  */
   static void determineForces(const QuaternionTransformationSystem::QuaternionVector &rotations, Eigen::ArrayXXd& velocities,
                               Eigen::ArrayXXd& accelerations, Eigen::ArrayXXd& forces,
                               const double dt, const double executionTime,
                               const double alphaZ = 25.0, const double betaZ = 6.25);


  /**Initializes the dmp from the given config file*/
  virtual bool initialize(const std::string& initialConfigPath);

  /**Initializes the dmp from the given model */
  virtual bool initialize(const QuaternionDmpModel& model);

  /**Configures the dmp from a yaml file.
  *  @see configure(QuaternionDmpConfig) */
  virtual bool configure(const std::string& configPath);

  /**Configures the dmp from a yaml string.
  *  @see configure(QuaternionDmpConfig)*/
  virtual bool configureYaml(const std::string& yaml);

  /**
  * @param[in] values array of size >= 4 that represents a quaternion. [w, x, ,y, z]
  *                   Only the first 4 elements of the array will be used.
  *
  * \note The inputs will be normalized
  */
  virtual void setInputs(const double *values, int numInputs);

  /**
  * @param[out] values array of size >= 4 that represents a quaternion. [w, x, ,y, z]
  *                    The first 4 elements will be overwritten with the output.
  */
  virtual void getOutputs(double *values, int numOutputs) const;

  virtual void step();

  virtual bool canStep() const;


  /**Applies the given configuration.
  *  This can be done at any time after initialize() has been called.
  *  The dmp can be reconfigured mid-run, e.g. to change the goal.
  *
  *  \note The start and end positions will be normalized. Therefore the values
  *        might differ from the values defined in the configuration file. */
  bool configure(const QuaternionDmpConfig& config);


  /**
  * Sets the weights matrix of the forcing term
  * Should be a 3xN matrix.
  * N should be equal to the number of centers in the function approximator.
  */
  virtual void setWeights(const Eigen::ArrayXXd& newWeights);


  /**
  * Gets the weights matrix of the forcing term
  * Should be a 3xN matrix.
  * N should be equal to the number of centers in the function approximator.
  */
  virtual const Eigen::MatrixXd& getWeights();

  //creates the module info needed by the lib manager.
  //without it the lib manager would be unable to load this module at run time.
  CREATE_MODULE_INFO();

private:

  std::auto_ptr<CanonicalSystem> cs;
  std::auto_ptr<RbfFunctionApproximator> rbf;
  std::auto_ptr<ForcingTerm> ft;
  std::auto_ptr<QuaternionTransformationSystem> ts;

  Eigen::Quaterniond startPos;
  Eigen::Quaterniond endPos;
  Eigen::Quaterniond currentPos; /**< set by setInputs() */
  Eigen::Array3d startVel;

  bool initialized;/**<If true initialize() has been called successfully */
  double currentPhase;
  unsigned currentPhaseIndex;
  bool stepPossible; /**<True if at least one more step is possible */
};
}
