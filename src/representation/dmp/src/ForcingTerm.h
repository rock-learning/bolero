/**
 * @file forcing_term.h
 * @author Arne Boeckmann (arne.boeckmann@dfki.de)
 */

#pragma once

#include "RbfFunctionApproximator.h"
#include <Eigen/Core>

namespace dmp {


/**
 *  An arbitrarily parameterizable term of the transformation system.
 *
 *  The parameters of the forcing term shape the trajectory of the DMP. This
 *  version of the forcing term has been described e. g. by [Pastor2009]_. The
 *  forcing term is the normalized output of radial basis functions multiplied
 *  with the current phase:
 *
 *  \f[
 *  f(s) = \frac{\sum_i \psi_i(s) \cdot w_i}{\sum_i \psi_i(s)} \cdot s
 *  \f]
 *  where \f$ \psi_i \f$ are radial basis functions, s is the phase variable
 *  and \f$ w_i \f$ is the weight corresponding to the i-th basis function.
 *
 */
class ForcingTerm {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**\param weights \see setWeights() */
  ForcingTerm(const RbfFunctionApproximator& functionApproximator, const Eigen::MatrixXd& weights);

  /**Creates an invalid instance. setWeights() needs to be called before the
   * instance can be used.*/
  ForcingTerm(const RbfFunctionApproximator& functionApproximator);

  /**
   * Sets the weights matrix.
   * Each row contains the weights for one task space dimension.
   * The number of columns should be equal to the number of centers in the function approximator.
   * E.g.:
   *    numTaskSpaceDimensions = 6
   *    numCenters = 10
   *    Resulting matrix is 6x10
   *    matrix[2][5] is the weight for the 6'th rbf in dimension 3
   *
   * @param newWeights
   */
  void setWeights(const Eigen::MatrixXd& newWeights);

  const Eigen::MatrixXd& getWeights() const;

  /**Calculates the value of the forcing term for each dimension at the specified phase */
  void calculateValue(const double phase, Eigen::ArrayXd& out) const;

  /**
   * Returns the activations of the function approximator for the given phase
   *
   * \param[in] s the phase
   * \param[in] normalized If true the activations will be normalized
   * \param[out] The activations
   */
  template <class Derived>
  void getActivations(const double s, const bool normalized, Eigen::ArrayBase<Derived>& out) const;


private:
  const RbfFunctionApproximator& functionApproximator;

  /**  Each row provides weights for one task space dimension.
   *   Number of weights is the same as numFeatures*/
  Eigen::MatrixXd weights;
};

template <class Derived>
void ForcingTerm::getActivations(const double s, const bool normalized,
                                 Eigen::ArrayBase<Derived>& out) const {
  if(normalized)
  {
    return functionApproximator.getNormalizedActivations(s, out);
  }
  else
  {
    return functionApproximator.getActivations(s, out);
  }
}

}


