#pragma once
#include "DmpBehavior.h"
#include "QuaternionDmp.h"
#include "RigidBodyDmpConfig.h"
#include <LoadableBehavior.h>
#include <lib_manager/LibInterface.hpp>
#include <string>
#include <memory>
#include <Eigen/Core>
#include "Dmp.h"

namespace dmp
{

/**
* A behavior that can be used to specify trajectories of 3d poses.
* A pose is the translation and orientation of an object in R^3.
* The translation is defined by a 3d vector.
* The rotation is defined by a quaternion.
*/
class RigidBodyDmp : public bolero::LoadableBehavior
{
public:

  RigidBodyDmp(lib_manager::LibManager *manager);

  /**
  * \param values Format: [p_x, p_y, p_z, v_x, v_y, v_z, a_x, a_y, a_z, w, x, y, z]
  *               p = position, v = velocity, a = acceleration
  *               [w, x, y, z] = rotation as quaternion
  */
  virtual void setInputs(const double *values, int numInputs);

  /**
  * \param[out] values Format: [p_x, p_y, p_z, v_x, v_y, v_z, a_x, a_y, a_z, w, x, y, z]
  *                            p = position, v = velocity, a = acceleration
  *                            [w, x, y, z] = rotation as quaternion
  */
  virtual void getOutputs(double *values, int numOutputs) const;

  virtual void step();

  virtual bool canStep() const;
  /**Initialize from yaml file */
  virtual bool initialize(const std::string& modelPath);
  /**Initialize from yaml string */
  virtual bool initializeYaml(const std::string yaml);
  /**Initialize from DMPModel */
  virtual bool initialize(const dmp_cpp::DMPModel& model);

 //FIXME right now reconfiguration is not possible. configure() can only be called once
  /*Configure from yaml file*/
  virtual bool configure(const std::string& configPath);
  /**Configure from yaml string */
  virtual bool configureYaml(const std::string& yaml);
  /**Configure from RigidBodyDmpConfig */
  virtual bool configure(const RigidBodyDmpConfig& config);

  /**
  * \param[in] positions A 3xN matrix of the demonstrated 3D positions.
  *                      Each column is one position.
  *                      The corresponding velocities and accelerations
  *                      will be approximated using the specified dt.
  *                      Providing velocities and accelerations from the outside
  *                      is possible but not implemented at the moment.
  * \param[in] positionRows The number of rows in the positions matrix.
  *                          Should always be 3.
  * \param[in] positionCols The number of columns in the positions matrix.
  * \param[in] rotations A 4xN matrix of the demonstrated 3D rotations.
  *                      Should be encoded as quaternions.
  *                      Each column is one rotation.
  *                      Row 0 = w
  *                      Row 1 = x
  *                      Row 2 = y
  *                      Row 3 = z
  * \param[in] rotationRows Number of rows in the rotation matrix. Should always be 4.
  * \param[in] rotationCols Number of columns in the rotation matrix. Should be
  *                         the same as columns in the position matrix.
  * \param[out] forces A 6xN matrix of forces. Should be allocated by the user.
  *                    Will be filled by this function.
  *                    Each column contains the forces for one data point.
  *                    The first 3 rows contain the forces for the positions,
  *                    the last 3 rows contain the forces for the rotations.
  * \param[in] forcesRows Should always be 6.
  * \param[in] forcesCols Should be the same as rotationCols and positionCols
  *
  * \note The pointers should use the same storage order as eigen, by default this
  *       is column-major. However it is possible to change eigens storage order
  *       using global defines. Check if this is the case with your project!!!
  *
  */
  static void determineForces(const double* positions, const int positionRows,
                              const int positionCols, const double* rotations,
                              const int rotationRows, const int rotationCols,
                              double* forces, const int forcesRows, const int forcesCols,
                              const double executionTime, const double dt,
                              const double alphaZ = 25.0, const double betaZ = 6.25);

  /** Same as above but expects all data in one matrix.
  *
  * \param[in] positions A 7XN array containing the translational and rotational
  *                      positions.
  *                      Each column should be one position. Rotations should be encoded
  *                      as quaternion.
  *                      Row 0 = x
  *                      Row 1 = y
  *                      Row 2 = z
  *                      Row 3 = rot_w
  *                      Row 4 = rot_x
  *                      Row 5 = rot_y
  *                      Row 6 = rot_u
  * \param[in] positionRows The number of rows in the positions matrix.
  *                          Should always be 7.
  * \param[in] positionCols The number of columns in the positions matrix.
  * \param[in] forcesRows Should always be 6.
  * \param[in] forcesCols Should be the same as positionCols
  * \param[out] forces A 6xN matrix of forces. Should be allocated by the user.
  *                    Will be filled by this function.
  *                    Each column contains the forces for one data point.
  *                    The first 3 rows contain the forces for the positions,
  *                    the last 3 rows contain the forces for the rotations.
  */
  static void determineForces(const double* positions, const int positionRows,
          const int positionCols, double* forces, const int forcesRows,
          const int forcesCols, const double executionTime, const double dt,
          const double alphaZ = 25.0, const double betaZ = 6.25);



  /**
  * Returns the activations of the function approximator for the given phase
  *
  * \param[in] s the phase
  * \param[in] normalized If true the activations will be normalized
  * \param[out] out The activations (will be resized if need be).
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
  void getActivations(const double s, const bool normalized, double* activations,
          const int size) const;

  /** Sets the weights.
  * @param rows should always be 6
  * @param cols should always be the same as the columns in centers and widths
  */
  void setWeights(const double* weights, const int rows, const int cols);

  /** Gets the weights.
  * @param rows should always be 6
  * @param cols should always be the same as the columns in centers and widths
  */
  void getWeights(double* weights, const int rows, const int cols);

  /**
   * \note The phases will be generated on the fly, therefore calling this method
   *       is computationally expensive.
   */
  virtual void getPhases(double* phases, const int len) const;


  //creates the module info needed by the lib manager.
  //without it the lib manager would be unable to load this module at run time.
  CREATE_MODULE_INFO();

private:
  lib_manager::LibManager* manager;
  std::auto_ptr<DmpBehavior> translationDmp;
  std::auto_ptr<QuaternionDmp> rotationDmp;
  bool initialized; /**<True if initialize() has been called successfully */
  bool configured; /**<True if configure() has been called successfully */
};

template <class Derived>
void RigidBodyDmp::getActivations(const double s, const bool normalized, Eigen::ArrayBase<Derived> &out) const
{
  assert(initialized);
  //note: This assumes that the translation and rotation dmp both use the same
  //      forcing term and parameters.
  translationDmp->getDmp().getActivations(s, normalized, out);
}

}
