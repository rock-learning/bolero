#include <promp.h>
#include "BasisFunctions.h"
#include "Trajectory.h"
#include <iostream>

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
      numBF_(numBF), numDim_(numDim), isStroke_(isStroke), overlap_(overlap) {}

void TrajectoryData::sampleTrajectoryData(TrajectoryData &traj) const
{
  Trajectory(*this).sampleTrajectoty().getData(traj);
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
  Trajectory(timestampsVector, valueVector, overlap_, numBF_).getData(*this);
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
  //std::vector<ConditionPoint> cp = ConditionPoint::fromMatrix(points_);
  //t.setConditions(cp);
  //t.getData(*this);
}

void CombinedTrajectoryData::addTrajectory(const TrajectoryData trajectory, const double *activation,
                                           const int numActivation)
{
  if (means_.empty())
  {
    numBF_ = trajectory.numBF_;
    numDim_ = trajectory.numDim_;
    isStroke_ = trajectory.isStroke_;
    overlap_ = trajectory.overlap_;
  }
  means_.push_back(trajectory.mean_);
  covariances_.push_back(trajectory.covariance_);
  activations_.push_back(std::vector<double>(activation, activation + numActivation));
}

void CombinedTrajectoryData::step(const double timestamp, double *values, int numValues, double *covs, int numCovs) const
{
  CombinedTrajectory traj(*this);

  if (numValues)
  {
    Matrix value(values, numValues, 1);
    value = traj.getValueMean(timestamp);
  }

  if (numCovs)
  {
    Matrix cov(covs, numValues, numValues);
    cov = traj.getValueCovars(timestamp);
  }
}

void CombinedTrajectoryData::getValues(const double *timestamps, const int numTimestamps, double *means, int numMeans,
                                       double *covars, int numCovars) const
{
  // const VectorXd timestamps_ = constVector(timestamps, numTimestamps);
  // Matrix means_(means, 2 * numDim_, numTimestamps);
  // std::vector<Matrix> covars_;
  // CombinedTrajectory trajectory(*this);
  // for (int i = 0; i < numTimestamps; i++)
  // {
  //   covars_.push_back(Matrix(covars + (2 * numDim_ * 2 * numDim_ * i), 2 * numDim_, 2 * numDim_));
  //   covars_[i] = trajectory.getValueCovars(timestamps_(i));
  // }
  // means_ = trajectory.getValueMean(timestamps_);
}

/*  
  void getTrajectory(int dimensions, double* weights, int numWeights, double* times, int numTimes, double* out, int numOut){
    Matrix weight(weights, dimensions, numWeights/dimensions);
    Vector time(times, numTimes);
    Matrix output(out, 2*dimensions,numOut/(2*dimensions));
    Eigen::MatrixXd tmp = output;
    //Trajectory(weight).getValueMean((Eigen::VectorXd)time,tmp);
    //output = tmp;
  }
  
  void getCovar(int dimensions, double* weights, int numWeights, double* times, int numTimes, double* out, int numOut){
    Matrix weight(weights, dimensions, numWeights/dimensions);
    Vector time(times, numTimes);
    Matrix output(out, 2*dimensions,numOut/(2*dimensions));
    Eigen::MatrixXd tmp = output;
   // Trajectory(weight).getValueMean((Eigen::VectorXd)time,tmp);
   // output = tmp;
  }
  
  void getWeights(double* demonstrations, int numDemonstrations, double* times, int numTimes, double* weights, int numWeights){
    Matrix demonstration(demonstrations,numTimes,numDemonstrations/numTimes);
    Matrix weight(weights,numDemonstrations/numTimes,numWeights/(numDemonstrations/numTimes));
    Vector time(times, numTimes);
    weight = StrokeBasisFunctions(weight.cols()).getValue(time).colPivHouseholderQr().solve((MatrixXd)demonstration).transpose();
  }
  
  void getStatistics(int dimensions, double* weights, int numWeights, double* means, int numMeans, double* covariances, int numCovariances){
    Vector mean(means,numMeans);
    Matrix  covariance(covariances, numMeans, numMeans);
    Matrix weight(weights,numWeights/numMeans,numMeans);
    
    
    mean = weight.colwise().mean();
    Eigen::MatrixXd centered = weight.rowwise() - weight.colwise().mean(); 
    covariance = ((centered.adjoint() * centered) / (weight.rows()-1.0));
  }
  
  
  void getStatistics2(int numWeights, int num, double* times,int numTimes, double* values, int numValues, double* means, int numMeans, double* covariances, int numCovariances){
	std::vector<VectorXd> timeVector(num); 
	std::vector<MatrixXd> valueVector(num);
	for(int i = 0; i < num; i++){
	  timeVector[i] = Vector(times+(numTimes/num*i),numTimes/num);
	  valueVector[i] = Matrix(values+(numValues/num*i),numTimes/num,numValues/numTimes);	  
	}
	Matrix mean(means,numValues/numTimes,numWeights);
        Matrix covariance(covariances, numValues/numTimes*numWeights,numValues/numTimes*numWeights);
	Trajectory t(timeVector,valueVector,numWeights);
	mean = t.getWeightMean();
	covariance = t.getWeightCovars(); 
    
  }
  
  void getTrajectory2(int dimensions, double* weights, int numWeights, double* covariances, int numCovariances,double* times, int numTimes, double* values, int numvalues,double* confi, int numConfi ){
    Vector weight(weights, numWeights);
    Matrix covariance(covariances, numWeights,numWeights);
    Vector time(times, numTimes);    
    Matrix value(values, 2*dimensions,numTimes);
    std::vector<Matrix> confiVector;//(numTimes);
    Trajectory trajectory(numWeights/dimensions, weight,covariance);
    for(int i = 0; i < numTimes; i++){
	  confiVector.push_back(Matrix(confi+(2*dimensions*2*dimensions*i),2*dimensions,2*dimensions));
	  Eigen::MatrixXd tmp = confiVector[i];
	  tmp.setZero();
	  //confiVector[i].setZero();
	  trajectory.getValueCovars(1,time(i),tmp);
	  confiVector[i] = tmp;
          //xconfiVector[i].setZero();	  
    }
    Eigen::MatrixXd tmp = value;
    trajectory.getValueMean((Eigen::VectorXd)time,tmp);
    value = tmp;
    
  }
  
  void getStatistics3(int num, double* times,int numTimes, double* values, int numValues, TrajectoryData& simple){
	std::vector<VectorXd> timeVector(num); 
	std::vector<MatrixXd> valueVector(num);
	for(int i = 0; i < num; i++){
	  timeVector[i] = Vector(times+(numTimes/num*i),numTimes/num);
	  valueVector[i] = Matrix(values+(numValues/num*i),numTimes/num,numValues/numTimes);	  
	}
	Trajectory t(timeVector,valueVector,simple.numBF);
	t.getData(simple);    
  }
  
  void getTrajectory3(double* times, int numTimes, double* values, int numvalues,double* confi, int numConfi, TrajectoryData& simple){
    Vector time(times, numTimes);    
    Matrix value(values, 2*simple.numDim,numTimes);
    std::vector<Matrix> confiVector;//(numTimes);
    Trajectory trajectory(simple);
    for(int i = 0; i < numTimes; i++){
	  confiVector.push_back(Matrix(confi+(2*simple.numDim*2*simple.numDim*i),2*simple.numDim,2*simple.numDim));
	  Eigen::MatrixXd tmp = confiVector[i];
	  tmp.setZero();
	  trajectory.getValueCovars(1,time(i),tmp);
	  confiVector[i] = tmp;
    }
    Eigen::MatrixXd tmp = value;
    trajectory.getValueMean((Eigen::VectorXd)time,tmp);
    value = tmp;
    
    
  }
  
  
*/
}