/**
 * @file DmpBehavior.cpp
 *  @author Arne Boeckmann (arne.boeckmann@dfki.de)
 */

#include "DmpBehavior.h"
#include <cassert>
#include "Dmp.h"

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
    expectedState(INITIALIZE), stepPossible(false)
{
    LoadableBehavior::init(1, 1);
}

bool DmpBehavior::initialize(const std::string& modelPath)
{
  model.from_yaml_file(modelPath, ""); //FIXME find a way to specify the model name
  return initialize(model);
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
  //note: configure() can be called multiple times during the execution
  //therefore we can only assert that the expectedState is != INITIALIZE
  assert(expectedState != INITIALIZE);
  if(NULL == dmp.get())
  {
    std::cerr << "DmpBehavior: configure() called on uninitialized Behavior." << std::endl;
    return false;
  }

  if(expectedState == CONFIGURE)
  {
    //this is the first time that configure() was called.
    //Do a complete initialization of the dmp
    if(config.is_valid())
    {
      dmp->initialize(config);

      //Create the internal data buffer
      setNumInputs(config.dmp_startPosition.size() + config.dmp_startVelocity.size()
                   + config.dmp_startAcceleration.size());

      setNumOutputs(config.dmp_endPosition.size() + config.dmp_endVelocity.size()
                    + config.dmp_endAcceleration.size());

      assert(getNumInputs() == getNumOutputs());
      assert(getNumInputs() > 0);

      data.resize(getNumInputs());
      expectedState = CONFIGURED; //now the dmp is ready to use
    }
    else
    {
      return false;
    }
  }
  else
  {
    //the dmp has been initialized before.

    //FIXME right now it is also not possible to change the execution time
    //Therefore we just change the goal. It is possible to change the start
    //in the first step.

    //only a partialy innitialized config is needed for reconfiguration
    bool changed = false;
    if(config.dmp_endPosition.size() >= 0 &&
       config.dmp_endVelocity.size() == config.dmp_endPosition.size() &&
       config.dmp_endAcceleration.size() == config.dmp_endPosition.size())
    {
      dmp->changeGoal(
          config.dmp_endPosition.data(), config.dmp_endVelocity.data(),
          config.dmp_endAcceleration.data(), config.dmp_endAcceleration.size());
      changed = true;
    }
    if(config.dmp_startPosition.size() >= 0 &&
       config.dmp_startVelocity.size() == config.dmp_startPosition.size() &&
       config.dmp_startAcceleration.size() == config.dmp_startPosition.size() &&
       expectedState == CONFIGURED)
    {
      //It is not possible to change the startPos/Vel/Acc
      //without resetting the phase.
      dmp->changeStart(
          config.dmp_startPosition.data(), config.dmp_startVelocity.data(),
          config.dmp_startAcceleration.data(), config.dmp_startAcceleration.size());
      changed = true;
    }
    return changed;
  }

  return true;
}

void DmpBehavior::setInputs(const double* values, int numInputs)
{
  assert(numInputs == data.size());
  assert(expectedState == SET_INPUTS || expectedState == CONFIGURED);

  //the const_cast is ok because we only read from the Map
  data = Map<ArrayXd>(const_cast<double*>(values), numInputs);//create a temporary map and copy the values

  expectedState = STEP;
}

bool DmpBehavior::canStep() const
{
  return stepPossible;
}

void  DmpBehavior::step()
{
  assert(expectedState == STEP);

  const int dim = dmp->getTaskDimensions();
  //note: executeStep() modifies the parameters, therefore they may not be r-values.
  //      For that reason the l-values pos, vel and acc are created.
  typedef Eigen::DenseBase<Eigen::Array<double, Dynamic, 1> >::SegmentReturnType Segment;
  Segment pos = data.segment(0, dim);
  Segment vel = data.segment(dim, dim);
  Segment acc = data.segment(dim + dim, dim);

  stepPossible = dmp->executeStep(pos, vel, acc);

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

bool DmpBehavior::initialize(const dmp_cpp::DMPModel &model)
{
  assert(expectedState == INITIALIZE);
  if(model.is_valid())
  {
    dmp.reset(new Dmp(model));
    expectedState = CONFIGURE;
    stepPossible = true;
    return true;
  }
  return false;
}

Dmp &DmpBehavior::getDmp()
{
  assert(dmp.get());
  return *(dmp.get());
}
} /* namespace dmp */


