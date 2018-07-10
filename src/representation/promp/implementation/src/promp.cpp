#include <promp.h>
#include "BasisFunctions.h"
#include "Trajectory.h"
#include <iostream>
#include <chrono>

namespace promp
{
typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Matrix;
typedef Eigen::Map<Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor>> Vector;
typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> constMatrix;
typedef Eigen::Map<const Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor>> constVector;

TrajectoryData::TrajectoryData(const int numBF, const int numDim, const bool isStroke, const double overlap)
    : mean_(
          std::vector<double>(numBF * numDim)),
      covariance_(std::vector<double>(numBF * numDim * numBF * numDim)),
      iterationLimit_(100),numBF_(numBF), numDim_(numDim), isStroke_(isStroke), overlap_(overlap),
      randomState_(time(0))  {}

void TrajectoryData::sampleTrajectoryData(TrajectoryData &traj)
{  
  Trajectory(*this).sampleTrajectoty(randomState_).getData(traj);
}

void TrajectoryData::stepCov(const double timestamp, double *covs, int numCovs) const
{
  Matrix cov(covs, std::sqrt(numCovs), std::sqrt(numCovs));
  cov = Trajectory(*this).getValueCovars(timestamp);
}

void TrajectoryData::step(const double timestamp, double *values, int numValues) const
{
  Matrix value(values, numValues, 1);
  value = Trajectory(*this).getValueMean(timestamp);
}

void TrajectoryData::imitate(const double *sizes, const int numSizes, const double *timestamps, const int numTimestamps,
                             const double *values,
                             const int numValues)
{
  const VectorXd sizes_ = constVector(sizes, numSizes);
  std::vector<VectorXd> timestampsVector;
  std::vector<MatrixXd> valueVector;
  size_t counter = 0;
  for (int i = 0; i < numSizes; i++)
  {
    timestampsVector.push_back(
        constVector(timestamps + counter, sizes_(i)));
    valueVector.push_back(
        constMatrix(values + counter * numDim_, sizes_(i),
                    numDim_));
    counter += sizes_(i);
  }
  Trajectory(timestampsVector, valueVector, overlap_, numBF_,iterationLimit_).getData(*this);
}

void TrajectoryData::getValues(const double *timestamps, const int numTimestamps, double *means, int numMeans,
                               double *covars, int numCovars) const
{
  const VectorXd timestamps_ = constVector(timestamps, numTimestamps);
  Matrix means_(means, 2 * numDim_, numTimestamps);
  Matrix covars_(covars,numTimestamps,2 * numDim_*2*numDim_);
  Trajectory trajectory(*this);

  covars_ = trajectory.getValueCovars(timestamps_);
  means_ = trajectory.getValueMean(timestamps_);
}

void TrajectoryData::condition(const int count, const double *points, const int numPoints)
{
  conditions_ = std::vector<double>(numPoints);
  memcpy(conditions_.data(),points,sizeof(double)*numPoints);
}

}