#include "RigidBodyDmp.h"
#include "QuaternionDmpModel.h"
#include "RigidBodyDmpConfig.h"
#include "Dmp.h"
#include <Eigen/Dense>


namespace dmp
{

using namespace bolero;
using namespace Eigen;
using namespace std;
using namespace dmp_cpp;
using namespace lib_manager;
DESTROY_LIB(RigidBodyDmp);
CREATE_LIB(RigidBodyDmp);

RigidBodyDmp::RigidBodyDmp(LibManager* manager) :
        LoadableBehavior(manager, "RigidBodyDmp", 1),
        manager(manager),
        translationDmp(),
        initialized(false),
        configured(false)
{
  LoadableBehavior::init(13, 13);
}

void RigidBodyDmp::setInputs(const double* values, int numInputs)
{
  assert(numInputs >= 13);
  assert(configured);
  //data format [p_x, p_y, p_z, v_x, v_y, v_z, a_x, a_y, a_z, w, x, y, z]
  translationDmp->setInputs(values, 9);
  rotationDmp->setInputs(&values[9], 4);
}

void RigidBodyDmp::getOutputs(double* values, int numOutputs) const
{
  assert(numOutputs >= 13);
  assert(configured);
  translationDmp->getOutputs(values, 9);
  rotationDmp->getOutputs(&values[9], 4);
}

void RigidBodyDmp::step()
{
  assert(configured);
  translationDmp->step();
  rotationDmp->step();
}

bool RigidBodyDmp::canStep() const
{
  return translationDmp->canStep() && rotationDmp->canStep(); // TODO
}


bool RigidBodyDmp::configure(const std::string& configPath)
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

bool RigidBodyDmp::configureYaml(const std::string& yaml)
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

bool RigidBodyDmp::initialize(const std::string& modelPath)
{
  DMPModel model(modelPath, "");
  return initialize(model);
}

bool RigidBodyDmp::initializeYaml(const std::string yaml)
{
  DMPModel model;
  if(model.from_yaml_string(yaml, ""))//FIXME find way to set name
  {
    return initialize(model);
  }
  return false;
}


bool RigidBodyDmp::configure(const RigidBodyDmpConfig &config)
{
  if(NULL != translationDmp.get() && NULL != rotationDmp.get() && initialized)
  {
    configured = rotationDmp->configure(config.rotationConfig) && translationDmp->configure(config.translationConfig);
    return configured;
  }
  else
  {
    return false;
  }
}

bool RigidBodyDmp::initialize(const dmp_cpp::DMPModel &model)
{
  // TODO
  /*@note  Some part of the code will only work correctly
  *        if the canonical systems of both dmps are initialized
  *        using the same parameters.*/
  if(!model.is_valid())
  {
   return false;
  }
  if(model.ft_weights.size() != 6)
  {
   return false; //3 rows for the positions, 3 for the rotation
  }
  DMPModel translationModel(model);
  //keep the first 3 dimensions, they are the weights for the position
  translationModel.ft_weights.resize(3);

  QuaternionDmpModel rotationModel(model);
  rotationModel.ft_weights.resize(3);
  rotationModel.ft_weights[0] = model.ft_weights[3];
  rotationModel.ft_weights[1] = model.ft_weights[4];
  rotationModel.ft_weights[2] = model.ft_weights[5];

  if(translationModel.is_valid() && rotationModel.is_valid())
  {
    translationDmp.reset(new DmpBehavior(manager));
    rotationDmp.reset(new QuaternionDmp(manager));

    initialized = rotationDmp->initialize(rotationModel) &&
            translationDmp->initialize(translationModel);
    return initialized;
  }
  else
  {
    return false;
  }
}

void RigidBodyDmp::setWeights(const double *weights, const int rows, const int cols)
{
  assert(initialized);
  assert(rows == 6);

  ArrayXXd weightsArr = Map<ArrayXXd>(const_cast<double*>(weights), rows, cols);
  translationDmp->setWeights(weightsArr.block(0, 0, 3, cols));
  rotationDmp->setWeights(weightsArr.block(3, 0, 3, cols));
}

void RigidBodyDmp::getWeights(double* weights, const int rows, const int cols)
{
  assert(initialized);
  assert(rows == 6);

  Map<ArrayXXd> weightsArr = Map<ArrayXXd>(weights, rows, cols);
  weightsArr.block(0, 0, 3, cols) = translationDmp->getWeights();
  weightsArr.block(3, 0, 3, cols) = rotationDmp->getWeights();
}

}//end namespace
