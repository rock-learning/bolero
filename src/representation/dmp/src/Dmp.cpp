#include "Dmp.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include <assert.h>
#include "EigenHelpers.h"


namespace dmp {
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::ArrayBase;
using Eigen::Map;

Dmp::Dmp(const dmp_cpp::DMPModel& model) :
    cs(model.cs_execution_time, model.cs_alpha, model.cs_dt),
    rbf(EigenHelpers::toEigen(model.rbf_centers), EigenHelpers::toEigen(model.rbf_widths)),
    ft(rbf, EigenHelpers::toEigen(model.ft_weights)),
    ts(ft, model.ts_tau, model.ts_dt, model.ts_alpha_z, model.ts_beta_z),
    currentPhase(1.0),
    currentPhaseIndex(0),
    weightsSet(true),
    taskDimensions(0),
    name(model.model_name),
    integrationSteps(4),
    initialized(false)
{
  assert(model.is_valid());
}

Dmp::Dmp(const dmp_cpp::DMPModel& model, const dmp_cpp::DMPConfig& config) :
    cs(model.cs_execution_time, model.cs_alpha, model.cs_dt),
    rbf(EigenHelpers::toEigen(model.rbf_centers), EigenHelpers::toEigen(model.rbf_widths)),
    ft(rbf, EigenHelpers::toEigen(model.ft_weights)),
    ts(ft, model.ts_tau, model.ts_dt, model.ts_alpha_z, model.ts_beta_z),
    currentPhase(1.0),
    currentPhaseIndex(0),
    weightsSet(true),
    taskDimensions(0),
    name(model.model_name),
    integrationSteps(4),
    initialized(false)
{//FIXME use delegate ctor upon switch to c++11
  assert(model.is_valid());
  assert(config.is_valid());
  initialize(config);
}


Dmp::Dmp(const double executionTime, const double alpha, const double dt,
         const unsigned numCenters, const double overlap, const double alphaZ,
         const double betaZ, const unsigned integrationSteps) :
    cs(executionTime, alpha, dt),
    //the rbf is always initialized using a cs with executionTime 1.0
    //to avoid inaccuracies when changing the execution time.
    //In theory the execution time should not have any influence on the phase,
    //however in practice it does have a slight influence due to numerical
    //inaccuracies.
    //If the rbf would be initialized using the given cs,
    //we would see a slight difference in results between a dmp that was
    //initialized with excutionTime t and one that was initialized with different
    //execution time but later changed to t.
    rbf(CanonicalSystem(1.0, alpha, 1.0 / (executionTime / dt)), numCenters, overlap),
    ft(rbf),
    ts(ft, executionTime, dt, alphaZ, betaZ),
    currentPhase(1.0),
    currentPhaseIndex(0),
    weightsSet(false),
    taskDimensions(0),
    integrationSteps(integrationSteps),
    initialized(false)
{}

Dmp::Dmp(const Dmp& other) :
    cs(other.cs),
    rbf(other.rbf),
    ft(rbf, other.ft.getWeights()),
    ts(ft, other.ts),
    currentPhase(1.0),
    currentPhaseIndex(0),
    weightsSet(other.weightsSet),
    taskDimensions(0),
    name(other.name),
    integrationSteps(other.integrationSteps),
    initialized(false)
{
  if(ft.getWeights().rows() > 0)
    taskDimensions = ft.getWeights().rows();
}


void Dmp::determineForces(const ArrayXXd& positions, ArrayXXd& velocities,
    ArrayXXd& accelerations, ArrayXXd& forces, const double executionTime,
    const double dt, const double alphaZ, const double betaZ,
    bool allowFinalVelocity)
{
  //positions.cols() == numPhases is required to be able to determine the forces
  assert(positions.cols() == (int)(executionTime / dt + 0.5) + 1);
  TransformationSystem::determineForces(positions, velocities,
                                        accelerations, forces, executionTime, dt,
                                        alphaZ, betaZ, allowFinalVelocity);
}

void Dmp::determineForces(const double* positions, double* velocities,
                          double* accelerations, const int posVelAccRows,
                          const int posVelAccCols, double* forces,
                          const int forcesRows, const int forcesCols,
                          const double executionTime, const double dt,
                          const double alphaZ, const double betaZ,
                          bool allowFinalVelocity)
{
  //This Method is not performance critical, therefore we just copy the data
  //The const_cast is ok because the map is only a temporary that is never modified
  ArrayXXd positionsArr = Map<ArrayXXd>(const_cast<double*>(positions), posVelAccRows, posVelAccCols);
  ArrayXXd accelArr;
  ArrayXXd velsArr;
  if(velocities != NULL)
  {
    velsArr = Map<ArrayXXd>(velocities, posVelAccRows, posVelAccCols);
  }
  if(accelerations != NULL)
  {
    accelArr = Map<ArrayXXd>(accelerations, posVelAccRows, posVelAccCols);
  }
  ArrayXXd forcesArr;
  Dmp::determineForces(positionsArr, velsArr, accelArr, forcesArr, executionTime,
                       dt, alphaZ, betaZ, allowFinalVelocity);
  Map<ArrayXXd>(forces, forcesRows, forcesCols) = forcesArr;
}


void Dmp::initialize(const ArrayXd& startPos, const ArrayXd& startVel,
    const ArrayXd& startAcc, const ArrayXd& endPos, const ArrayXd& endVel,
    const ArrayXd& endAcc)
{
  assert(startPos.size() == startVel.size());
  assert(startPos.size() == startAcc.size());
  assert(startPos.size() == endPos.size());
  assert(startPos.size() == endVel.size());
  assert(startPos.size() == endAcc.size());

  if(weightsSet)
  {
    //depending on which ctor has been used the weights may or may not have been set.
    //this assert only makes sense if the weights have been set.
    //the same assert is also present in setWeights().
    //there has to be one row of weights for each dimension.
    assert(startPos.size() == ft.getWeights().rows());
  }

  assertNonNanInf(startPos);
  assertNonNanInf(startVel);
  assertNonNanInf(startAcc);
  assertNonNanInf(endPos);
  assertNonNanInf(endVel);
  assertNonNanInf(endAcc);

  ts.initialize(startPos, startVel, startAcc, endPos, endVel, endAcc);
  currentPhase = 1.0;
  currentPhaseIndex = 0;

  taskDimensions = startPos.size();

  this->startPos = startPos;
  this->startVel = startVel;
  this->startAcc = startAcc;
  this->endPos = endPos;
  this->endVel = endVel;
  this->endAcc = endAcc;
  initialized = true;
}

void Dmp::initialize(const dmp_cpp::DMPConfig& config) {
  assert(config.is_valid());

  initialize(EigenHelpers::toEigen(config.dmp_startPosition),
             EigenHelpers::toEigen(config.dmp_startVelocity),
             EigenHelpers::toEigen(config.dmp_startAcceleration),
             EigenHelpers::toEigen(config.dmp_endPosition),
             EigenHelpers::toEigen(config.dmp_endVelocity),
             EigenHelpers::toEigen(config.dmp_endAcceleration));
  changeTime(config.dmp_execution_time);
}

void Dmp::initialize(const double* startPos, const double* startVel,
                     const double* startAcc, const double* endPos,
                     const double* endVel, const double* endAcc, const int len)
{
  //there is no need to use Eigen::Map or any other
  //template magic here because initialize is not a performance
  //critical method.
  this->startPos.resize(len);
  this->startVel.resize(len);
  this->startAcc.resize(len);
  this->endPos.resize(len);
  this->endVel.resize(len);
  this->endAcc.resize(len);

  for(int i = 0; i < len; ++i)
  {
    this->startPos[i] = startPos[i];
    this->startVel[i] = startVel[i];
    this->startAcc[i] = startAcc[i];
    this->endPos[i] = endPos[i];
    this->endVel[i] = endVel[i];
    this->endAcc[i] = endAcc[i];
  }

  initialize(this->startPos, this->startVel, this->startAcc,
             this->endPos, this->endVel, this->endAcc);

}


void Dmp::changeGoal(const ArrayXd& position, const ArrayXd& velocity,
    const ArrayXd& acceleration)
{
  //dimensionality is asserted inside changeGoal()
  ts.changeGoal(position, velocity, acceleration);
  endPos = position;
  endVel = velocity;
  endAcc = acceleration;
}

void Dmp::changeGoal(const double* position, const double* velocity,
                     const double* acceleration, const unsigned len)
{
  //dimensionality may not change without reinitialization
  assert(len == endPos.size());
  for(unsigned i = 0; i < len; ++i)
  {
    endPos[i] = position[i];
    endVel[i] = velocity[i];
    endAcc[i] = acceleration[i];
  }
  ts.changeGoal(endPos, endVel, endAcc);
}

void Dmp::changeStart(const double* position, const double* velocity,
                      const double* acceleration, const unsigned len)
{
  //dimensionality may not change without reinitialization
  assert(len == endPos.size());
  for(unsigned i = 0; i < len; ++i)
  {
    startPos[i] = position[i];
    startVel[i] = velocity[i];
    startAcc[i] = acceleration[i];
  }
  ts.changeStart(startPos, startVel, startAcc);
}

void Dmp::changeTime(const double newTime)
{
  cs = CanonicalSystem(newTime, cs.getAlpha(), cs.getDt());
  ts.setExecutionTime(newTime);
}

bool Dmp::executeStep(double* position, double* velocity,
                      double* acceleration, const int len)
{
  assert(len == startPos.size());
  Map<ArrayXd> posMap(position, len);
  Map<ArrayXd> velMap(velocity, len);
  Map<ArrayXd> accMap(acceleration, len);
  return executeStep(posMap, velMap, accMap);
}

void Dmp::setWeights(const ArrayXXd& newWeights)
{
  if(initialized) //startPos has only been set if initialize has been called.
  {
    //there has to be one row of weights for each dimension.
    //the same assert can be found in initialize, therefore it does not matter whether
    //initialize or setWeights is called first
    assert(startPos.size() == newWeights.rows());
  }
  ft.setWeights(newWeights.matrix());
  weightsSet = true;
  taskDimensions = newWeights.rows();
}

void Dmp::setWeights(const double* newWeights, const int rows, const int cols)
{
  //the const_cast is ok because we only need the map
  //to copy the data into the array
  double* pWeights = const_cast<double*>(newWeights);
  const ArrayXXd weights = Map<ArrayXXd>(pWeights, rows, cols);
  setWeights(weights);
}

void Dmp::getWeights(double* weights, const int rows, const int cols) const
{
  const MatrixXd& ftWeights = ft.getWeights();
  assert(rows == ftWeights.rows());
  assert(cols == ftWeights.cols());
  Map<ArrayXXd> map(weights, rows, cols);
  map = ftWeights;
}

const Eigen::MatrixXd& Dmp::getWeights()
{
    return ft.getWeights();
}

void Dmp::getActivations(const double s, const bool normalized, double* activations,
                    const int size) const {
  const int numCenters = rbf.getCenterCount();
  assert(size >= numCenters);
  Map<ArrayXd> map(activations, size);
  getActivations(s, normalized, map);
}



void Dmp::getPhases(ArrayXd& phases) const
{
  //the resize is done here because cs.getPhases takes
  //an ArrayBases as parameter instead of ArrayXd to be able
  //to cope with Eigen::Map as well.
  //ArrayBase does not allow for resize, therefore resize before
  //calling
  phases.resize(cs.getNumberOfPhases());
  cs.getPhases(phases);
}

void Dmp::getPhases(double* phases, const int len) const
{
  assert((unsigned)len == cs.getNumberOfPhases());
  Map<ArrayXd> phasesMap(phases, len);
  cs.getPhases(phasesMap);
}

double Dmp::getCurrentPhase() const
{
    return currentPhase;
}

double Dmp::getDt() const{
    return cs.getDt();
}

int Dmp::getTaskDimensions()
{
    return taskDimensions;
}

dmp_cpp::DMPModel Dmp::generateModel() const
{
  dmp_cpp::DMPModel model;
  model.cs_execution_time = cs.getExecutionTime();
  model.cs_alpha = cs.getAlpha();
  model.cs_dt = cs.getDt();
  model.rbf_centers = EigenHelpers::toStdVec(rbf.getCenters());
  model.rbf_widths = EigenHelpers::toStdVec(rbf.getWidths());

  assert(weightsSet);
  model.ft_weights = EigenHelpers::toStdVec(ft.getWeights());

  model.ts_alpha_z = ts.getAlphaZ();
  model.ts_beta_z = ts.getBetaZ();
  model.ts_dt = ts.getDt();
  model.ts_tau = ts.getTau();
  model.model_name = name;

  return model;
}

dmp_cpp::DMPConfig Dmp::generateConfig() const
{
  assert(initialized);//config data is only available if the dmp has been initalized

  dmp_cpp::DMPConfig config;
  config.config_name = name;
  config.dmp_execution_time = ts.getTau();
  config.dmp_startPosition = EigenHelpers::toStdVec(startPos);
  config.dmp_endPosition = EigenHelpers::toStdVec(endPos);
  config.dmp_startVelocity = EigenHelpers::toStdVec(startVel);
  config.dmp_endVelocity = EigenHelpers::toStdVec(endVel);
  config.dmp_startAcceleration = EigenHelpers::toStdVec(startAcc);
  config.dmp_endAcceleration = EigenHelpers::toStdVec(endAcc);
  return config;
}
void Dmp::assertNonNanInf(const ArrayXd& data) const
{
  for(unsigned i = 0; i < data.size(); ++i)
  {
    assert(!isnan(data[i]));
    assert(!isinf(data[i]));
  }
}
}
