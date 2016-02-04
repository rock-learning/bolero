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
  return translationDmp->canStep() && rotationDmp->canStep();
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

void RigidBodyDmp::determineForces(const double *positions, const int positionRows,
                                   const int positionCols, const double *rotations,
                                   const int rotationRows, const int rotationCols,
                                   double *forces, const int forcesRows,
                                   const int forcesCols, const double executionTime,
                                   const double dt, double const alphaZ, double const betaZ)
{
  assert(positionCols > 0);
  assert(positionRows == 3);
  assert(rotationCols == positionCols);
  assert(rotationRows == 4);
  assert(forcesCols == positionCols);
  assert(forcesRows == 6);

  //This Method is not performance critical, therefore we just copy the data a few times

  //The const_cast is ok because the map is only a temporary that is never modified
  ArrayXXd positionsArr = Map<ArrayXXd>(const_cast<double*>(positions), positionRows, positionCols);
  assert(positionsArr.allFinite());
  ArrayXXd velsArr(0, 0); //will be filled by Dmp::determineForces
  ArrayXXd accArr(0, 0); //will be filled by Dmp::determineForces


  ArrayXXd positionForces(positionRows, positionCols);
  Dmp::determineForces(positionsArr, velsArr, accArr, positionForces, executionTime,
                       dt, alphaZ, betaZ);

  ArrayXXd rotationsArr = Map<ArrayXXd>(const_cast<double*>(rotations), rotationRows, rotationCols);
  assert(rotationsArr.allFinite());
  QuaternionTransformationSystem::QuaternionVector rotationsVector;
  for(int i = 0; i < rotationCols; ++i)
  {
    Quaterniond q;
    q.w() = rotationsArr.col(i)(0);
    q.x() = rotationsArr.col(i)(1);
    q.y() = rotationsArr.col(i)(2);
    q.z() = rotationsArr.col(i)(3);
    q.normalize();//has to be done to avoid nans
    rotationsVector.push_back(q);
  }

  ArrayXXd rotationVelocities(0,0);
  ArrayXXd rotationAccelerations(0,0);
  ArrayXXd rotationForces(rotationRows, rotationCols);
  QuaternionDmp::determineForces(rotationsVector, rotationVelocities, rotationAccelerations, rotationForces,
                                 dt, executionTime, alphaZ, betaZ);

  ArrayXXd forcesArr(6, positionCols);
  forcesArr.setConstant(NAN); //initially set forces to nan, this way we can assert afterwards that all forces have been calculated correctly
  forcesArr.block(0, 0, 3, forcesCols) = positionForces;
  forcesArr.block(3, 0, 3, forcesCols) = rotationForces;
  assert(forcesArr.allFinite());

  Map<ArrayXXd>(forces, forcesRows, forcesCols) = forcesArr;
}

void RigidBodyDmp::determineForces(const double *positions, const int positionRows, const int positionCols,
                                   double *forces, const int forcesRows, const int forcesCols,
                                   const double executionTime, const double dt, double const alphaZ, double const betaZ)
{
  assert(positionRows == 7);
  assert(forcesCols == positionCols);
  assert(forcesRows == 6);
  //This Method is not performance critical, therefore we just copy the data a few times
  ArrayXXd positionsArr = Map<ArrayXXd>(const_cast<double*>(positions), positionRows, positionCols);
  assert(positionsArr.allFinite());
  ArrayXXd translations = positionsArr.block(0, 0, 3, positionCols);

  ArrayXXd velsArr(0, 0); //will be filled by Dmp::determineForces
  ArrayXXd accArr(0, 0); //will be filled by Dmp::determineForces

  ArrayXXd positionForces(3, positionCols);
  Dmp::determineForces(translations, velsArr, accArr, positionForces, executionTime,
                       dt, alphaZ, betaZ);

  ArrayXXd rotations = positionsArr.block(3, 0, 4, positionCols);

  QuaternionTransformationSystem::QuaternionVector rotationsVector;
  for(int i = 0; i < positionCols; ++i)
  {
    Quaterniond q;
    q.w() = rotations.col(i)(0);
    q.x() = rotations.col(i)(1);
    q.y() = rotations.col(i)(2);
    q.z() = rotations.col(i)(3);
    q.normalize();//has to be done to avoid nans
    rotationsVector.push_back(q);
  }

  ArrayXXd rotationVelocities(0,0);
  ArrayXXd rotationAccelerations(0,0);
  ArrayXXd rotationForces(3, positionCols);
  QuaternionDmp::determineForces(rotationsVector, rotationVelocities, rotationAccelerations, rotationForces,
          dt, executionTime, alphaZ, betaZ);

  ArrayXXd forcesArr(6, positionCols);
  forcesArr.setConstant(NAN); //initially set forces to nan, this way we can assert afterwards that all forces have been calculated correctly
  forcesArr.block(0, 0, 3, forcesCols) = positionForces;
  forcesArr.block(3, 0, 3, forcesCols) = rotationForces;
  assert(forcesArr.allFinite());

  Map<ArrayXXd>(forces, forcesRows, forcesCols) = forcesArr;
}

void RigidBodyDmp::getActivations(const double s, const bool normalized,
                                  double *activations, const int size) const
{
  assert(initialized);
  translationDmp->getDmp().getActivations(s, normalized, activations, size);
}

void RigidBodyDmp::setWeights(const double *weights, const int rows, const int cols)
{
  assert(initialized);
  assert(rows == 6);

  ArrayXXd weightsArr = Map<ArrayXXd>(const_cast<double*>(weights), rows, cols);
  translationDmp->getDmp().setWeights(weightsArr.block(0, 0, 3, cols));
  rotationDmp->setWeights(weightsArr.block(3, 0, 3, cols));
}

void RigidBodyDmp::getWeights(double* weights, const int rows, const int cols)
{
  assert(initialized);
  assert(rows == 6);

  Map<ArrayXXd> weightsArr = Map<ArrayXXd>(weights, rows, cols);
  weightsArr.block(0, 0, 3, cols) = translationDmp->getDmp().getWeights();
  weightsArr.block(3, 0, 3, cols) = rotationDmp->getWeights();
}

void RigidBodyDmp::getPhases(double *phases, const int len) const
{
  assert(translationDmp.get());
  //note: this works as long as both the translation and the rotation
  //      dmp use the same canonical system.
  //      Right now this is always the case. If it ever changes this code needs
  //      to change as well.
  assert(rotationDmp.get());
  translationDmp->getDmp().getPhases(phases, len);
}
}//end namespace
