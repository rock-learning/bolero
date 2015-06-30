#include "RbfFunctionApproximator.h"
#include <assert.h>


namespace dmp {
using Eigen::ArrayXd;
using Eigen::Map;

 RbfFunctionApproximator::RbfFunctionApproximator(const CanonicalSystem& canonicalSystem,
         const int numCenters, const double overlap) {
    calculateCenters(canonicalSystem, numCenters, overlap, centers, widths);
  }

 RbfFunctionApproximator::RbfFunctionApproximator(const ArrayXd& centers,
                                                  const ArrayXd& widths) :
    centers(centers), widths(widths) {}

 RbfFunctionApproximator::RbfFunctionApproximator(const RbfFunctionApproximator& other) :
        centers(other.centers), widths(other.widths) {}


  int RbfFunctionApproximator::getCenterCount() const {
    return centers.size();
  }

  ArrayXd RbfFunctionApproximator::getCenters() const {
    return centers;
  }

  ArrayXd RbfFunctionApproximator::getWidths() const {
    return widths;
  }

void RbfFunctionApproximator::calculateCenters(const double lastPhaseValue,
        const double executionTime, const double dt, const unsigned numCenters,
        const double overlap, Eigen::ArrayXd &centers, Eigen::ArrayXd &widths)
{
  const double alpha = CanonicalSystem::calculateAlpha(lastPhaseValue, dt, executionTime);
  CanonicalSystem cs(executionTime, alpha, dt);
  calculateCenters(cs, numCenters, overlap, centers, widths);
}

void RbfFunctionApproximator::calculateCenters(const CanonicalSystem &canonicalSystem,
        const int numCenters, const double overlap, Eigen::ArrayXd &centers, Eigen::ArrayXd &widths)
{
  centers.resize(numCenters);
  widths.resize(numCenters);
  const double step = canonicalSystem.getExecutionTime() / (numCenters - 1); //-1 because we want the last entry to be getExecutionTime
  const double logOverlap = -std::log(overlap);
  //do first iteration outside loop because we need access to i and i-1 in loop
  double time = 0.0;
  centers(0) = canonicalSystem.getPhase(time);
  for(int i = 1; i < numCenters; ++i)
  {
    //alternatively Eigen::LinSpaced can be used, however it does exactly the same calculation
    time = i*step; //normally lower_border+i*step but lower_border is 0
    centers(i) = canonicalSystem.getPhase(time);
    //Choose width of RBF basis functions automatically so that the
    //RBF centered at one center has value overlap at the next center
    const double diff = centers(i)-centers(i-1);
    widths(i-1) = logOverlap / (diff*diff);
  }
  //width of last gausian cannot be calculated, just use the same width as the one before
  widths(numCenters-1) = widths(numCenters -2);
}


void RbfFunctionApproximator::calculateCenters(const double lastPhaseValue,
        const double executionTime, const double dt, const unsigned numCenters,
        const double overlap, double *centers, double *widths)
{
  assert(numCenters > 0);
  ArrayXd centersArr;
  ArrayXd widthsArr;
  calculateCenters(lastPhaseValue, executionTime, dt, numCenters, overlap, centersArr,
                   widthsArr);
  Map<ArrayXd>(widths, numCenters) = widthsArr;
  Map<ArrayXd>(centers, numCenters) = centersArr;
}
}//end namespace
