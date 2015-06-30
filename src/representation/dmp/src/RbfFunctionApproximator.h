/**
 * @file function_approximator.h
 *
 * Collection of several function approximators that can be used inside the
 * forcing term.
 * All approximators follow the same interface, however they do not actually
 * implement any interface to avoid the virtual function overhead.
 *
 * @author Arne Boeckmann (arne.boeckmann@dfki.de)
 */

#pragma once

#include "CanonicalSystem.h"
#include <Eigen/Core>

namespace dmp {


/**
 *
 * An object of this class provides a mapping from phase to feature vector.
 * The feature vector has sum 1. The mapping is obtained based on radial basis
 * functions (RBFs). The centers of the RBFs are chosen uniformly time domain
 * and projected to the phase space.

 */
class RbfFunctionApproximator {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW


  /**
   * Distributes the RBFs in a way that the RBF at center i has value \p overlap
   * at centers (i + 1) and (i - 1).
   */
  RbfFunctionApproximator(const CanonicalSystem& canonicalSystem, const int numCenters,
                                  const double overlap);
  /**
   * Distributes the RBFs according the the specified centers and widths.
   * \param centers Center locations of the RBFs
   * \param widths Widths of the RBFs. widths[i] is the width of the RBF at centers[i]
   */
  RbfFunctionApproximator(const Eigen::ArrayXd& centers, const Eigen::ArrayXd& widths);


  RbfFunctionApproximator(const RbfFunctionApproximator& other);

  /**Returns the activations for a given phase */
  template <class Derived>
  void getActivations(const double s, Eigen::ArrayBase<Derived>& out) const;

  /**Returns the number of centers */
  virtual int getCenterCount() const;

  virtual Eigen::ArrayXd getCenters() const;

  virtual Eigen::ArrayXd getWidths() const;

  template <class Derived>
  void getNormalizedActivations(const double s, Eigen::ArrayBase<Derived>& out) const;

  static void calculateCenters(const double lastPhaseValue, const double executionTime,
                               const double dt, const unsigned numCenters,
                               const double overlap, Eigen::ArrayXd& centers,
          Eigen::ArrayXd& widths);

  static void calculateCenters(const CanonicalSystem& canonicalSystem, const int numCenters,
                               const double overlap, Eigen::ArrayXd& centers,
                               Eigen::ArrayXd& widths);

  /**
  * @param centers An array of size numCenters. Should be allocated by the caller,
  *                will be filled by this function.
  * @param widths An array of size numCenters. Should be allocated by the caller,
  *                will be filled by this function.
  */
  static void calculateCenters(const double lastPhaseValue, const double executionTime,
          const double dt, const unsigned numCenters,
          const double overlap, double* centers,
          double* widths);

private:
  Eigen::ArrayXd centers; /**<Center location of each gausian */
  Eigen::ArrayXd widths; /**<Width of each gausian */

};

template <class Derived>
void RbfFunctionApproximator::getActivations(const double s, Eigen::ArrayBase<Derived>& out) const {
  assert(s <= 1 && s >= 0);
  out = (-widths * (s - centers).pow(2)).exp();
  /**
   * NOTE:
   * For a very large number of centers it might be possible to calculate only the subset
   * of activations around the given phase instead of all activations.
   * Activations that are very far away from the current phase usually have very little
   * influence.
   */

}

template <class Derived>
void RbfFunctionApproximator::getNormalizedActivations(const double s, Eigen::ArrayBase<Derived>& out) const {
  getActivations(s, out);
  const double sum = out.sum();
  out /= sum;
}


}
