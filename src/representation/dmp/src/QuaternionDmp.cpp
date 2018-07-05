#include "QuaternionDmp.h"
#include "QuaternionDmpModel.h"
#include "QuaternionDmpConfig.h"
#include <assert.h>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <Eigen/Core>
#include "EigenHelpers.h"
#include <Dmp.h>


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
        initialized(false)
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
  if(model.is_valid() && model.ft_weights.size() == 3)
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

    lastT = 0.0;
    t = 0.0;

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
    startT = 0.0;
    goalT = config.executionTime;

    Eigen::VectorXd q0 = EigenHelpers::toEigen(config.startPosition);
    q0 /= q0.norm();
    Eigen::VectorXd q0d = EigenHelpers::toEigen(config.startVelocity);
    Eigen::VectorXd g = EigenHelpers::toEigen(config.endPosition);
    g /= g.norm();

    startData.resize(10);
    goalData.resize(10);
    lastData.resize(10);
    data.resize(10);

    setNumInputs(10);
    setNumOutputs(10);

    assert(4 == q0.size());
    assert(3 == q0d.size());
    assert(4 == g.size());

    EigenHelpers::assertNonNanInf(q0);
    EigenHelpers::assertNonNanInf(q0d);
    EigenHelpers::assertNonNanInf(g);

    startData.segment(0, 4) = q0;
    startData.segment(4, 3) = q0d;
    startData.segment(7, 3).setZero();

    goalData.segment(0, 4) = g;
    goalData.segment(4, 3).setZero();
    goalData.segment(7, 3).setZero();

    if(t <= 0.0)
    {
      lastData.segment(4, 3) = q0d;
      lastData.segment(7, 3).setZero();
    }

    return true;
  }
  else
  {
    return false;
  }
}

void QuaternionDmp::setInputs(const double *values, int numInputs)
{
  assert(initialized);
  assert(numInputs >= 4); //quaternions always have 4 elements

  lastData.segment<4>(0) = Map<const Array<double, 4, 1> >(values, 4);
  lastData.segment<4>(0) /= lastData.segment<4>(0).matrix().norm();
}

void QuaternionDmp::getOutputs(double *values, int numOutputs) const
{
  assert(initialized);
  assert(numOutputs >= 4);

  Map<Array<double, 4, 1> > map(values);
  map = data.segment<4>(0);

  // TODO That should be easier. Why does segment not work on the left side?
  Map<ArrayXd>((double*) (lastData.data() + 4), 6) = data.segment<6>(4);
}

void QuaternionDmp::step()
{
  assert(initialized);

  Dmp::quaternionDmpStep(
    lastT, t,
    lastData.data(), 4,
    lastData.data() + 4, 3,
    lastData.data() + 7, 3,
    data.data(), 4,
    data.data() + 4, 3,
    data.data() + 7, 3,
    goalData.data(), 4,
    goalData.data() + 4, 3,
    goalData.data() + 7, 3,
    startData.data(), 4,
    startData.data() + 4, 3,
    startData.data() + 7, 3,
    goalT, startT,
    weights.data(), weights.cols(), weights.rows(),
    widths.data(), widths.rows(),
    centers.data(), centers.rows(),
    alphaY, betaY, alphaZ, 0.001
  );

  lastT = t;
  t += dt;
}

bool QuaternionDmp::canStep() const
{
  return t <= goalT;
}

void QuaternionDmp::setWeights(const Eigen::ArrayXXd &weights)
{
  assert(weights.rows() == 3);
  this->weights = weights;
}

const Eigen::MatrixXd QuaternionDmp::getWeights()
{
  return weights.matrix();
}
}//end namespace
