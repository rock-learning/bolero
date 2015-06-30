#include "ForcingTerm.h"
#include <cassert>

namespace dmp
{
using Eigen::MatrixXd;
using Eigen::ArrayXd;

ForcingTerm::ForcingTerm(const RbfFunctionApproximator& functionApproximator) :
                         functionApproximator(functionApproximator){}

ForcingTerm::ForcingTerm(const RbfFunctionApproximator& functionApproximator, const MatrixXd& weights) :
    functionApproximator(functionApproximator), weights(weights) {}

void ForcingTerm::setWeights(const MatrixXd& newWeights) {
  assert(newWeights.cols() == functionApproximator.getCenterCount());
  weights = newWeights;
}

const MatrixXd& ForcingTerm::getWeights() const {
  return weights;
}

/**Calculates the value of the forcing term for each dimension of the specified phase */
void ForcingTerm::calculateValue(const double phase, ArrayXd& out) const {
  assert(weights.size() > 0);
  ArrayXd activations;
  functionApproximator.getNormalizedActivations(phase, activations);
  out = phase * weights * activations.matrix();
}
} //end namespace


