/**
 * @file Dmp.h
 * @author Arne Böckmann (arne.boeckmann@dfki.de)
 */
#pragma once
#include "CanonicalSystem.h"
#include "ForcingTerm.h"
#include "RbfFunctionApproximator.h"
#include "TransformationSystem.h"
#include "ForcingTerm.h"
#include "DMPModel.h"
#include "DMPConfig.h"

#include <Eigen/Core>
#include <string>


namespace dmp {

class Dmp{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * Create a Dmp from a saved model.
   * If you want to execute it you need to call initialize() beforehand.
   */
  Dmp(const dmp_cpp::DMPModel& model);

  /**
   * Create a Dmp from a saved model and initialize it using the
   * specified config
   */
  Dmp(const dmp_cpp::DMPModel& model, const dmp_cpp::DMPConfig& config);


  /**
   * Creates a partially initialized dmp.
   * This dmp can be used for learning but not for execution.
   * If you want to execute it you need to call setWeights()
   * and initialize() beforehand.
   *
   * \param executionTime The execution time of the dmp. Should be divisible by dt.
   * \param alpha Canonical system phase decay parameter (see CanonicalSystem for details)
   * \param dt Time between two steps.
   * \param numCenters Number of centers used in the rbf.
   * \param overlap Overlap between the rbf centers (See RbfFunctionApproximator for details)
   * \param alphaZ Constant that is used to achieve critical dampening.
   * \param betaZ Constant that is used to achieve critical dampening.
   *              The dmp is critically damped as long as betaZ = alphaZ / 4
   * \param integrationSteps Internally the dmp subdivides every integration step
   *                         into n smaller steps to avoid integration errors.
   *                         This parameter defines how many integration steps
   *                         will be done for one 'real step'
   *
   */
  Dmp(const double executionTime, const double alpha, const double dt,
      const unsigned numCenters, const double overlap, const double alphaZ = 25.0,
      const double betaZ = 6.25, const unsigned integrationSteps = 4);

  Dmp(const Dmp& other);

  virtual ~Dmp(){}

  dmp_cpp::DMPModel generateModel() const;

  dmp_cpp::DMPConfig generateConfig() const;


  /** Determines the forces needed to imitate the given trajectory.
   *
   * \param[in] positions The positions of the trajectory. Each column contains
   *                 one n-dimensional position.
   *                 positions.cols() should be floor(tau/dt) - 1
   * \param[in,out] velocities The velocities of the trajectory. Will be approximated
   *                   if empty (size == 0).
   * \param[in,out] accelerations The accelerations of the trajectory. Will be approximated
   *                   if empty (size == 0).
   * \param[out] forces The resulting forces.
  *  \param[in] executionTime Execution time of the whole trajectory
  *  \param[in] dt Time step between two data points
   * \note  This is essentially a wrapper for TransformationSystem::determineForces().
   *        Therefore detailed documentation of this function can be found there.
   */
  static void determineForces(const Eigen::ArrayXXd& positions, Eigen::ArrayXXd& velocities,
                              Eigen::ArrayXXd& accelerations, Eigen::ArrayXXd& forces,
                              const double executionTime, const double dt,
                              const double alphaZ = 25.0, const double betaZ = 6.25);

  /**
   * Same as above but accepts raw pointers.
   * \note If velocities or accelerations are NULL they will be approximated
   *       based on the positions to calculate the forces.
   *       However the approximations will not be returned.
   *       If you want to get the approximated values please use the Eigen version
   *       of this method (see above).
   * \note This function uses column major storage (just like Eigen).
   *       All parameters should be in column major storage order.
   *       The return value will als beo column major.
   */
  static void determineForces(const double* positions, double* velocities,
                              double* accelerations, const int posVelAccRows,
                              const int posVelAccCols, double* forces,
                              const int forcesRows, const int forcesCols,
                              const double executionTime, const double dt,
                              const double alphaZ = 25.0, const double betaZ = 6.25);


  /**
   * Sets the initial and final positions of the DMP and resets the phase
   * to 1.0.
   * This method should be called before calling executeStep() for the first
   * time.
   * This method can also be used to reset the DMP.
   *
   * \param startPos The starting position of the DMP.
   * \param startVel The starting velocity of the DMP.
   * \param startAcc The stating acceleration of the DMP.
   * \param endPos The end position of the DMP.
   * \param endVel The end velocity of the DMP.
   * \param endAcc The end acceleration of the DMP.
   *
   * \note All parameters should have the same dimensionality
   */
  virtual void initialize(const Eigen::ArrayXd& startPos, const Eigen::ArrayXd& startVel,
                          const Eigen::ArrayXd& startAcc, const Eigen::ArrayXd& endPos,
                          const Eigen::ArrayXd& endVel, const Eigen::ArrayXd& endAcc);
  /**
   * Same as above but uses raw pointers.
   */
  virtual void initialize(const double* startPos, const double* startVel,
                          const double* startAcc, const double* endPos,
                          const double* endVel, const double* endAcc, const int len);



  /**
   * Sets the initial and final positions as well as the executionTime
   * based on the specified config.
   * \note Resets the phase to 1.0
   */
  virtual void initialize(const dmp_cpp::DMPConfig& config);

  /**
   * Changes the end position of the DMP during run time.
   * \note This WILL NOT RESET the phase. If you want to reinitialize the DMP
   *       call initialize() instead.
   * \note This method will change the 'initial state'.
   *       I.e. if you call changeTime() after calling changeGoal()
   *       the goal will not be reset to the value given to initialize()
   *       but to the value given to changeGoal().
   * \note This method can only be called after initialize() has been called
   */
  virtual void changeGoal(const Eigen::ArrayXd& position, const Eigen::ArrayXd& velocity,
                          const Eigen::ArrayXd& acceleration);
  /**
   * Same as above but accepts raw pointer data
   */
  virtual void changeGoal(const double* position, const double* velocity,
                          const double* acceleration, const unsigned len);
  /**
   * Changes the start position of the DMP during run time.
   * \note This WILL NOT RESET the phase. If you want to reinitialize the DMP
   *       call initialize() instead.
   * \note This method will change the 'initial state'.
   *       I.e. if you call changeTime() after calling changeStart()
   *       the goal will not be reset to the value given to initialize()
   *       but to the value given to changeStart().
   * \note This method can only be called after initialize() has been called
   */
  virtual void changeStart(const double* position, const double* velocity,
                          const double* acceleration, const unsigned len);

  /**
   * Changes the execution time of the DMP.
   * \note This RESETS the dmp to the initial state.
   *       Currently it is not possible to change the execution time without
   *       resetting the DMP.
   *
   * \note Changing the execution time without changing dt results in
   *       a changed number of steps
   */
  virtual void changeTime(const double executionTime);

  /**
   * Execute a step of the DMP, i.e. move from one phase to the next.
   * After initialization the DMP is already at the first phase,
   * thus this method can be called \f$n-1\f$ times for a trajectory with
   * \f$n\f$ phases.
   * \param[in,out] position The current position. Will be overwritten with the new position.
   * \param[out] velocity The new velocity.
   * \param[out] acceleration the new acceleration.
   *
   * \return False if the DMP has reached the end of the trajectory, True otherwise.
   *         E.g. you can use while(executeStep(...)) to run every step of the
   *         DMP.
   *
   * \note The dmp can only be executed if the weights have been set.
   * \note The method is generic to be able to call it with Eigen::Map as well as ArrayXd
   *       or any other compatible type.
   */
  template <class DerivedA, class DerivedB, class DerivedC>
  bool executeStep(Eigen::ArrayBase<DerivedA>& position, Eigen::ArrayBase<DerivedB>& velocity,
                   Eigen::ArrayBase<DerivedC>& acceleration);

  /**
   * Same as above but accepts raw c arrays
   * \param len length of the arrays
   */
  virtual bool executeStep(double* position, double* velocity,
                           double* acceleration, const int len);

  /**
  * Sets the weights matrix of the forcing term
  * Each row contains the weights for one task space dimension.
  * The number of columns should be equal to the number of centers in the
  * function approximator. I.e.:
  *    numTaskSpaceDimensions = 6
  *    numCenters = 10
  *    Resulting matrix is 6x10
  *    matrix[2][5] is the weight for the 6'th rbf in dimension 3
  *
  * @param newWeights
  */
  virtual void setWeights(const Eigen::ArrayXXd& newWeights);

  /**
   * Same as above but accepts raw c arrays
   * \param newWeights a matrix
   * \param rows Number of rows of the matrix
   * \param cols Number of columns of the matrix
   * \note The data should be stored in column major format.
   */
  virtual void setWeights(const double* newWeights, const int rows, const int cols);

  /**
   * Returns the weights of the forcing term
   *
   * \param[out] weights the weights of the forcing term
   * \param rows Number of rows of the matrix
   * \param cols Number of columns of the matrix
   */
  virtual void getWeights(double* weights, const int rows, const int cols) const;

  /**
   * Returns the weights of the forcing term
   *
   * Each row contains the weights for one task space dimension.
   * The number of columns should be equal to the number of centers in the
   * function approximator. I.e.:
   *    numTaskSpaceDimensions = 6
   *    numCenters = 10
   *    Resulting matrix is 6x10
   *    matrix[2][5] is the weight for the 6'th rbf in dimension 3
   *
   * @return weights
   */
  virtual const Eigen::MatrixXd& getWeights();

  /**
   * Returns the activations of the function approximator for the given phase
   *
   * \param[in] s the phase
   * \param[in] normalized If true the activations will be normalized
   * \param[out] out The activations
   */
  template <class Derived>
  void getActivations(const double s, const bool normalized, Eigen::ArrayBase<Derived>& out) const;

  /**
   * Same as above but accepts raw c arrays
   * \param[out] activations array that the activations will be written to.
   * \param[in] size size of the activations arrays (has to be at least numCenters).
   *
   * \note If activations is bigger than numCenters the remaining elements
   *       will not be changed.
   */
  virtual void getActivations(const double s, const bool normalized, double* activations,
                              const int size) const;

  /**
   * \param[out] the phases
   *
   * \note The phases will be generated on the fly, therefore calling this method
   *       is computationally expensive.
   */
  virtual void getPhases(Eigen::ArrayXd& phases) const;

  /**
   * Same as above but uses raw array to output the phases.
   * \note len needs to be == number of phases, otherwise this
   *       method will crash
   */
  virtual void getPhases(double* phases, const int len) const;

  /**
   * \param[out] the current phase
   */
  virtual double getCurrentPhase() const;

  /**
   * \param[out] the dt
   */
  virtual double getDt() const;

  /**
   * \param[out] the number of task space dimensions
   */
  virtual int getTaskDimensions();

private:
  /**executes a step of the dmp using the given dt. This method can be used
   * to execute steps that do not use the default dt. */
  template <class DerivedA, class DerivedB, class DerivedC>
  void executeStep(const double dt, Eigen::ArrayBase<DerivedA>& position,
                           Eigen::ArrayBase<DerivedB>& velocity, Eigen::ArrayBase<DerivedC>& acceleration);

  void assertNonNanInf(const Eigen::ArrayXd& data) const;
                           
  CanonicalSystem cs;
  RbfFunctionApproximator rbf;
  ForcingTerm ft;
  TransformationSystem ts;
  double currentPhase;
  unsigned currentPhaseIndex;
  bool weightsSet; /**<True if weights have been set */
  int taskDimensions;
  std::string name; /**<Optional name of this dmp (only initialized from model) */
  /**The number of integration steps to do each execution step.
   * Increasing this number increases integration accuracy and decreases
   * performance.*/
  unsigned integrationSteps;
  bool initialized; /**<True if initialize() has been called */

  /**These arrays are used to store the values given to initialize() and changeGoal().
   * They are needed to be able to reset the DMP. */
  Eigen::ArrayXd startPos;
  Eigen::ArrayXd startVel;
  Eigen::ArrayXd startAcc;
  Eigen::ArrayXd endPos;
  Eigen::ArrayXd endVel;
  Eigen::ArrayXd endAcc;
};

template <class DerivedA, class DerivedB, class DerivedC>
bool Dmp::executeStep(Eigen::ArrayBase<DerivedA>& position, Eigen::ArrayBase<DerivedB>& velocity,
                      Eigen::ArrayBase<DerivedC>& acceleration)
{
  assert(weightsSet); //dmp can only be executed if weights have been set

  //to increase accuracy the Dmp does multiple small integration steps
  //instead of one large step
  const double dt = cs.getDt() / integrationSteps;
  for(unsigned i = 0; i < integrationSteps; ++i)
  {
    executeStep(dt, position, velocity, acceleration);
  }

  ++currentPhaseIndex;
  return currentPhaseIndex < cs.getNumberOfPhases() - 1;
}

template <class DerivedA, class DerivedB, class DerivedC>
void Dmp::executeStep(const double dt, Eigen::ArrayBase<DerivedA>& position,
                      Eigen::ArrayBase<DerivedB>& velocity, Eigen::ArrayBase<DerivedC>& acceleration)
{
  const double currentTime = cs.getTime(currentPhase);
  ts.executeStep(dt, currentPhase, currentTime, position, velocity, acceleration);
  currentPhase = cs.getPhase(currentTime + dt);
}

template <class Derived>
void Dmp::getActivations(const double s, const bool normalized,
    Eigen::ArrayBase<Derived>& out) const {
  ft.getActivations(s, normalized, out);
}


}



