/**
 * @file DmpBehavior.cpp
 *  @author Arne Boeckmann (arne.boeckmann@dfki.de)
 */

#include "DmpBehavior.h"
#include <cassert>
#include "Dmp.h"
#include "EigenHelpers.h"

namespace dmp {

using namespace bolero;
using namespace Eigen;
using namespace std;

//if the dmp behavior is used as part of another behavior the create and destroy
//functions should not be generated because the other behavior will generate them
//and there can only be one pair of create/destroy methods per library.
#ifdef BUILD_DMP_BEHAVIOR_STANDALONE
  DESTROY_LIB(DmpBehavior);
  CREATE_LIB(DmpBehavior);
#endif

DmpBehavior::DmpBehavior(lib_manager::LibManager *manager) :
    LoadableBehavior(manager, "DmpBehavior", 1), //NOTE: numInputs and numOutputs are set later when configure() is called
    expectedState(INITIALIZE)
{
    LoadableBehavior::init(1, 1);
}

bool DmpBehavior::initialize(const std::string& modelPath)
{
  model.from_yaml_file(modelPath, ""); //FIXME find a way to specify the model name
  return initialize(model);
}

bool DmpBehavior::initialize(const dmp_cpp::DMPModel &model)
{
  assert(expectedState == INITIALIZE);
  if(model.is_valid())
  {
    name = model.model_name;

    startT = 0.0;
    goalT = model.cs_execution_time;

    alphaY = model.ts_alpha_z;
    betaY = model.ts_beta_z;
    alphaZ = model.cs_alpha;

    dt = model.cs_dt;
    assert(model.ts_dt == dt);
    assert(model.ts_tau == goalT);

    centers = EigenHelpers::toEigen(model.rbf_centers);
    widths = EigenHelpers::toEigen(model.rbf_widths);

    Dmp::initializeRbf(
        widths.data(), widths.rows(), centers.data(), centers.rows(),
        goalT, startT, 0.8, alphaZ
    );

    weights = EigenHelpers::toEigen(model.ft_weights);

    taskDimensions = weights.cols();

    expectedState = CONFIGURE;

    return true;
  }
  return false;
}

bool DmpBehavior::configure(const std::string& configPath)
{
  if(config.from_yaml_file(configPath, "")) //FIXME find a way to specify the config name
  {
    return configure(config);
  }
  return false;
}

bool DmpBehavior::configureYaml(const std::string& yaml)
{
  //note: The return value of from_yaml_string is ignored because
  //      the config does not need to be fully valid in case of 
  //      reconfiguration.
  config.from_yaml_string(yaml, ""); //FIXME find a way to specify the config name
  return configure(config);
}


bool DmpBehavior::configure(const dmp_cpp::DMPConfig& config)
{
  // NOTE configure() can be called multiple times during the execution.
  // Therefore we can only assert that the expectedState is != INITIALIZE.
  assert(expectedState != INITIALIZE);

  if(config.is_valid())
  {
    startT = 0.0;
    goalT = config.dmp_execution_time;

    Eigen::ArrayXd y0 = EigenHelpers::toEigen(config.dmp_startPosition);
    Eigen::ArrayXd y0d = EigenHelpers::toEigen(config.dmp_startVelocity);
    Eigen::ArrayXd y0dd = EigenHelpers::toEigen(config.dmp_startAcceleration);
    Eigen::ArrayXd g = EigenHelpers::toEigen(config.dmp_endPosition);
    Eigen::ArrayXd gd = EigenHelpers::toEigen(config.dmp_endVelocity);
    Eigen::ArrayXd gdd = EigenHelpers::toEigen(config.dmp_endAcceleration);

    if(expectedState == CONFIGURE)
    {
      // This is the first time that configure() was called.
      // Do a complete initialization of the DMP.
      taskDimensions = y0.rows();
      assert(taskDimensions > 0);

      startData.resize(3 * taskDimensions);
      goalData.resize(3 * taskDimensions);
      lastData.resize(3 * taskDimensions);
      data.resize(3 * taskDimensions);

      setNumInputs(3 * taskDimensions);
      setNumOutputs(3 * taskDimensions);
    }
    else
    {
      assert(y0.size() == taskDimensions);
    }

    assert(taskDimensions == y0.size());
    assert(taskDimensions == y0d.size());
    assert(taskDimensions == y0dd.size());
    assert(taskDimensions == g.size());
    assert(taskDimensions == gd.size());
    assert(taskDimensions == gdd.size());

    EigenHelpers::assertNonNanInf(y0);
    EigenHelpers::assertNonNanInf(y0d);
    EigenHelpers::assertNonNanInf(y0dd);
    EigenHelpers::assertNonNanInf(g);
    EigenHelpers::assertNonNanInf(gd);
    EigenHelpers::assertNonNanInf(gdd);

    startData.segment(0 * taskDimensions, taskDimensions) = y0;
    startData.segment(1 * taskDimensions, taskDimensions) = y0d;
    startData.segment(2 * taskDimensions, taskDimensions) = y0dd;

    goalData.segment(0 * taskDimensions, taskDimensions) = g;
    goalData.segment(1 * taskDimensions, taskDimensions) = gd;
    goalData.segment(2 * taskDimensions, taskDimensions) = gdd;

    if(expectedState == CONFIGURE)
    {
      lastData.segment(0 * taskDimensions, taskDimensions) = y0;
      lastData.segment(1 * taskDimensions, taskDimensions) = y0d;
      lastData.segment(2 * taskDimensions, taskDimensions) = y0dd;

      lastT = 0.0;
      t = 0.0;

      expectedState = CONFIGURED;
      // Now the dmp is ready to use
    }
  }
  else
  {
    return false;
  }

  return true;
}

void DmpBehavior::setInputs(const double* values, int numInputs)
{
  assert(numInputs == data.size());
  assert(expectedState == SET_INPUTS || expectedState == CONFIGURED);

  //the const_cast is ok because we only read from the Map
  lastData = Map<ArrayXd>(const_cast<double*>(values), numInputs); // Create a temporary map and copy the values

  expectedState = STEP;
}

bool DmpBehavior::canStep() const
{
  return t <= goalT;
}

void  DmpBehavior::step()
{
  assert(expectedState == STEP);

  Dmp::dmpStep(
    lastT, t,
    lastData.data(), taskDimensions,
    lastData.data() + taskDimensions, taskDimensions,
    lastData.data() + 2 * taskDimensions, taskDimensions,
    data.data(), taskDimensions,
    data.data() + taskDimensions, taskDimensions,
    data.data() + 2 * taskDimensions, taskDimensions,
    goalData.data(), taskDimensions,
    goalData.data() + taskDimensions, taskDimensions,
    goalData.data() + 2 * taskDimensions, taskDimensions,
    startData.data(), taskDimensions,
    startData.data() + taskDimensions, taskDimensions,
    startData.data() + 2 * taskDimensions, taskDimensions,
    goalT, startT,
    weights.data(), weights.cols(), weights.rows(),
    widths.data(), widths.rows(),
    centers.data(), centers.rows(),
    alphaY, betaY, alphaZ, 0.001
  );

  lastT = t;
  t += dt;

  expectedState = GET_OUTPUTS;
}

void DmpBehavior::getOutputs(double* values, int numOutputs) const
{
  assert(numOutputs == data.size());
  assert(expectedState == GET_OUTPUTS);

  Map<ArrayXd> map(values, numOutputs);
  map = data;

  expectedState = SET_INPUTS;
}

} /* namespace dmp */


