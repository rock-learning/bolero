#include "QuaternionDmp.h"
#include "QuaternionDmpModel.h"
#include "QuaternionDmpConfig.h"
#include <assert.h>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <Eigen/Core>
#include "EigenHelpers.h"


namespace dmp
{

using namespace Eigen;

//if the quaternion dmp is used as part of another behavior the create and destroy
//functions should not be generated because the other behavior will generate them
//and there can only be one pair of create/destroy methods per library.
#ifdef BUILD_QUATERNION_STANDALONE
  DESTROY_LIB(QuaternionDmp);
  CREATE_LIB(QuaternionDmp);
#endif

QuaternionDmp::QuaternionDmp(lib_manager::LibManager* manager) :
        LoadableBehavior(manager, "QuaternionDmpBehavior", 1),
        startPos(Quaterniond::Identity()),
        endPos(Quaterniond::Identity()),
        currentPos(Quaterniond::Identity()),
        startVel(Array3d::Zero()),
        initialized(false),
        currentPhase(1.0),
        currentPhaseIndex(0),
        stepPossible(false)
{
  //all behaviors must use lazy initialization, therefore the constructor
  //does nearly nothing
  LoadableBehavior::init(4, 4);
}

bool QuaternionDmp::initialize(const std::string &initialConfigPath)
{
  QuaternionDmpModel model;
  if(!model.from_yaml_file(initialConfigPath, ""))//FIXME find way to set name
  {
    return false;
  }
  return initialize(model);
}

bool QuaternionDmp::initialize(const QuaternionDmpModel &model)
{
  if(model.is_valid())
  {
    // TODO

    currentPhase = 1.0;
    currentPhaseIndex = 0;
    initialized = true;
    return true;
  }
  else
  {
    return false;
  }
}

bool QuaternionDmp::configureYaml(const std::string &yaml)
{
  if(config.fromYamlString(yaml, ""))//FIXME find way to set name
  {
    return configure(config);
  }
  else
  {
    return false;
  }
}


bool QuaternionDmp::configure(const std::string &configPath)
{
  if(config.fromYamlFile(configPath, ""))//FIXME find way to set name
  {
    return configure(config);
  }
  else
  {
    return false;
  }
}

bool QuaternionDmp::configure(const QuaternionDmpConfig &config)
{
  if(initialized && config.isValid())
  {
    startPos = Quaterniond(config.startPosition[0], config.startPosition[1],
            config.startPosition[2], config.startPosition[3]);
    endPos = Quaterniond(config.endPosition[0], config.endPosition[1],
            config.endPosition[2], config.endPosition[3]);
    startVel << config.startVelocity[0], config.startVelocity[1],
            config.startVelocity[2];
    startPos.normalize();
    endPos.normalize();

    // TODO

    stepPossible = true;
    return true;
  }
  else
  {
    return false;
  }
}

void QuaternionDmp::setInputs(const double *values, int numInputs)
{
  assert(numInputs >= 4); //quaternions always have 4 elements
  currentPos = Quaterniond(values[0], values[1], values[2], values[3]);
  currentPos.normalize();
}

void QuaternionDmp::getOutputs(double *values, int numOutputs) const
{
  assert(numOutputs >= 4);
  values[0] = currentPos.w();
  values[1] = currentPos.vec().x();
  values[2] = currentPos.vec().y();
  values[3] = currentPos.vec().z();
}

void QuaternionDmp::step()
{
  assert(initialized);
  // TODO
}

bool QuaternionDmp::canStep() const
{
  return stepPossible;
}

void QuaternionDmp::setWeights(const Eigen::ArrayXXd &weights)
{
  assert(weights.rows() == 3);
  // TODO
}

const Eigen::MatrixXd& QuaternionDmp::getWeights()
{
  // TODO
}
}//end namespace
