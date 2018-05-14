#include "Trajectory.h"
#include "promp.h"
#include <iostream>

#include<Eigen/Core>
#include<Eigen/SVD>
#include<cmath>

#include <random>

using namespace promp;
int example_counter = 0;

typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > Matrix;
typedef Eigen::Map<Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor> > Vector;
typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > constMatrix;
typedef Eigen::Map<const Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor> > constVector;

std::vector<ConditionPoint> ConditionPoint::fromMatrix(MatrixXd pointMatrix) {
  std::vector<ConditionPoint> ret;

  for (int i = 0; i < pointMatrix.rows(); i++) {
    ret.push_back(ConditionPoint());
    ret.back().timestamp = pointMatrix(i, 0);
    ret.back().dimension = pointMatrix(i, 1);
    ret.back().derivative = pointMatrix(i, 2);
    ret.back().mean = pointMatrix(i, 3);
    ret.back().variance = pointMatrix(i, 4);
  }

  return ret;
}

Trajectory::Trajectory(const TrajectoryData &data) :
        numWeights_(data.numBF_), numDim_(data.numDim_), overlap_(data.overlap_) {
  type_ = data.isStroke_ ? Stroke : Periodic;
  weightMean_ = constVector(data.mean_.data(), data.numDim_ * data.numBF_);
  weightCovars_ = constMatrix(data.covariance_.data(), data.numDim_ * data.numBF_, data.numDim_ * data.numBF_);
  conditionPoints_ = ConditionPoint::fromMatrix(constMatrix(data.conditions_.data(), data.conditions_.size()/ConditionPoint::NUM_FIELDS, ConditionPoint::NUM_FIELDS));
  setBF();
  condition(weightMean_,weightCovars_);
}

Trajectory::Trajectory(const int numWeights, const VectorXd &weights, const double overlap, const MatrixXd &covars,
                       const TrajectoryType type)
        : numWeights_(numWeights), numDim_(weights.size() / numWeights), overlap_(overlap), weightMean_(weights),
          weightCovars_(covars),
          type_(type) {
  setBF();
}


Trajectory::Trajectory(const std::vector<VectorXd> &timestamps, const std::vector<MatrixXd> &values,
                       const double overlap, int numWeights,
                       const TrajectoryType type)
        : numWeights_(numWeights), numDim_(values.front().cols()), overlap_(overlap), type_(type) {
  setBF();
  imitate(timestamps, values);
}

void Trajectory::imitate(const std::vector<VectorXd> &timestamps, const std::vector<MatrixXd> &values) {

  beginTimeMeasure();

  weightMean_ = VectorXd::Zero(numDim_ * numWeights_);
  weightCovars_ = MatrixXd::Identity(numDim_ * numWeights_, numDim_ * numWeights_);
  standardDev_ = 1.;

  MatrixXd means(values.size(), numDim_ * numWeights_);
  std::vector<MatrixXd> covs(values.size(), MatrixXd::Identity(numFunc_ * numWeights_, numFunc_ * numWeights_));

  std::vector<MatrixXd> H(values.size());
  for (size_t i = 0; i < values.size(); i++) {
    MatrixXd H_partial = MatrixXd::Identity(values[i].rows(), values[i].rows());

    for (int y = 0; y < values[i].rows(); y++) {
      for (int preview = 1; preview < 2 && (y + preview) < values[i].rows(); preview++) {
        H_partial(y, y + preview) = -std::pow(0.7, preview);
      }
    }

    H[i] = (MatrixXd::Zero(values[i].size(), values[i].size()));
    for (int j = 0; j < numDim_; j++)
      H[i].block(H_partial.rows() * j, H_partial.rows() * j, H_partial.rows(), H_partial.rows()) = H_partial;
  }

  // value restructuring
  std::vector<MatrixXd> val(values.size());

  for (size_t i = 0; i < values.size(); i++) {
    val[i] = MatrixXd(values[i].size(), 1);
    for (int j = 0; j < numDim_; j++)
      val[i].block(timestamps[i].size() * j, 0, timestamps[i].size(), 1) = values[i].block(0, j, timestamps[i].size(),
                                                                                           1);
  }

  std::vector<MatrixXd> BF(values.size());

  for (size_t i = 0; i < values.size(); i++) {
    BF[i] = basisFunctions_->getValue(timestamps[i], numDim_).transpose();
  }

  std::vector<MatrixXd> R(values.size());

  for (size_t i = 0; i < values.size(); i++) {
    R[i] = H[i] * val[i];
  }

  std::vector<MatrixXd> RR(values.size());

  for (size_t i = 0; i < values.size(); i++) {
    RR[i] = R[i].transpose() * R[i];
  }

  std::vector<MatrixXd> BH(values.size());

  for (size_t i = 0; i < values.size(); i++) {
    BH[i] = BF[i] * H[i].transpose();
  }

  // calculation of mean factor for eStep
  std::vector<MatrixXd> mean_eStep(values.size());

  for (size_t i = 0; i < values.size(); i++) {
    mean_eStep[i] = (BH[i] * R[i]);
  }

  // calculation of cov factor for eStep
  std::vector<MatrixXd> cov_eStep(values.size());

  for (size_t i = 0; i < values.size(); i++) {
    cov_eStep[i] = (BH[i] * BH[i].transpose());
  }

  int sampleCount = 0;
  for (size_t i = 0; i < values.size(); i++) {
    sampleCount += values[i].rows();
  }

  std::cout << "Preparation Duration: " << endTimeMeasure() << "s" << std::endl;

  int counter = 0;
  beginTimeMeasure();
  VectorXd weightMean_old;
  MatrixXd weightCovars_old;
  do {
    counter++;
    weightMean_old = weightMean_;
    weightCovars_old = weightCovars_;

    for (size_t i = 0; i < values.size(); i++) {
      E_Step(1. / standardDev_ * mean_eStep[i], 1. / standardDev_ * cov_eStep[i], means.row(i), covs[i]);
    }

    M_Step(means, covs);

    M_Step2(means, covs, RR, mean_eStep, cov_eStep, sampleCount);
    //std::cout  <<((weightMean_old - weightMean_).norm() + (weightCovars_old - weightCovars_).norm()) << std::endl;;
  } while (counter < 100 &&
  ((weightMean_old - weightMean_).norm() + (weightCovars_old - weightCovars_).norm()) > 0.000001);

  std::cout << "Duration: " << endTimeMeasure() << "s" << std::endl;;
  //std::cout << "Iterations: " << counter << std::endl << std::endl;

}


MatrixXd Trajectory::getWeightMean() const {
  return weightMean_;
};

MatrixXd Trajectory::getWeightCovars() const {
  return weightCovars_;
};

MatrixXd Trajectory::getValueMean(const VectorXd &time) const {
  //beginTimeMeasure();
  MatrixXd out(numDim_ *numFunc_, time.size());
  
  MatrixXd bf = basisFunctions_->getValue(time);
  MatrixXd bfd = basisFunctions_->getValueDeriv(time);
  //std::cout << "basisF: " << endTimeMeasure() << std::endl;
  for (int dimension = 0; dimension < numDim_; dimension++) {
    out.row(numFunc_ * dimension) = bf * weightMean_.segment(numWeights_ * dimension, numWeights_);
    out.row((numFunc_ * dimension) + 1) = bfd * weightMean_.segment(numWeights_ * dimension, numWeights_);
  }
  
  return out;
}

VectorXd Trajectory::getValueMean(const double time) const {
  VectorXd timeVec(1);
  timeVec << time;
  return getValueMean(timeVec).col(0);
}

MatrixXd Trajectory::getValueCovars(const VectorXd &time) const {
  MatrixXd out(time.size(),numDim_*numFunc_*numDim_*numFunc_);
  MatrixXd value = basisFunctions_->getValue(time);
  MatrixXd valueDeriv = basisFunctions_->getValueDeriv(time);
  
  for (int y = 0; y < numDim_; y++) {
    for (int x = 0; x < numDim_; x++) {
      
      out.transpose().row((numFunc_ * y*4)+ (numFunc_ * x)) = (value *
                                             weightCovars_.block(y * numWeights_, x * numWeights_, numWeights_,
                                                                 numWeights_) *
                                             value.transpose()).diagonal();
      
      out.transpose().row((numFunc_ * y*4)+ (numFunc_ * x) + 1) = (value *
                                                 weightCovars_.block(y * numWeights_, x * numWeights_, numWeights_,
                                                                     numWeights_) *
                                                 valueDeriv.transpose()).diagonal();
      
      out.transpose().row((numFunc_ * y*4) + (numFunc_ * x)+4) = (valueDeriv *
                                                 weightCovars_.block(y * numWeights_, x * numWeights_, numWeights_,
                                                                     numWeights_) *
                                                 value.transpose()).diagonal();

      out.transpose().row((numFunc_ * y*4) + (numFunc_ * x) + 5) = (valueDeriv *
                                                     weightCovars_.block(y * numWeights_, x * numWeights_, numWeights_,
                                                                         numWeights_) *
                                                     valueDeriv.transpose()).diagonal();      
    }
  } 
  return out;
}

VectorXd Trajectory::getValueCovars(const double time) const {
  VectorXd timeVec(1);
  timeVec << time;
  return getValueCovars(timeVec).row(0);
}

void Trajectory::condition(VectorXd& weightMean,MatrixXd& weightCovars) const{
  if( conditionPoints_.empty())
    return;
  MatrixXd basisFunc_tmp = MatrixXd::Zero(conditionPoints_.size(), numDim_ * numWeights_);
  VectorXd means(conditionPoints_.size());
  VectorXd variances(conditionPoints_.size());

  for (size_t i = 0; i < conditionPoints_.size(); i++) {
    const ConditionPoint &point = conditionPoints_[i];
    std::cout << point.timestamp << ", " << point.dimension << ", " << point.derivative << ", " << point.mean << ", " << point.variance <<  std::endl;
    
    if (point.derivative == 0) {
      basisFunc_tmp.block(i, numWeights_ * point.dimension, 1, numWeights_).row(0) = basisFunctions_->getValue(
              point.timestamp).row(0);
    } else {
      basisFunc_tmp.block(i, numWeights_ * point.dimension, 1, numWeights_).row(
              0) = basisFunctions_->getValueDeriv(point.timestamp).row(0);
    }

    means(i) = point.mean;
    variances(i) = point.variance;
  }

  MatrixXd basisFunc = basisFunc_tmp.transpose();

  MatrixXd cov = variances.asDiagonal();
  cov = (cov + (basisFunc.transpose() * weightCovars * basisFunc)).inverse();

  VectorXd weightMeanNew =
          weightMean + weightCovars * basisFunc * cov * (means - (basisFunc.transpose() * weightMean));
  MatrixXd weightCovarsNew =
          weightCovars - (weightCovars * basisFunc * cov * basisFunc.transpose() * weightCovars);
  weightMean = weightMeanNew;
  weightCovars = weightCovarsNew;

}

void Trajectory::getData(TrajectoryData &data) const {
  std::memcpy(data.mean_.data(), weightMean_.data(), weightMean_.size() * sizeof(double));
  std::memcpy(data.covariance_.data(), weightCovars_.data(), weightCovars_.size() * sizeof(double));
}

Trajectory Trajectory::sampleTrajectoty() const {
  MatrixXd A = weightCovars_.selfadjointView<Lower>().llt().matrixL();
  VectorXd z(weightMean_.size());
  std::default_random_engine generator(time(0));
  std::normal_distribution<double> normalDist(0, 1);

  for (int i = 0; i < weightMean_.size(); i++) {
    z(i) = normalDist(generator);
  }

  VectorXd newMean = weightMean_ + (A * z);
  MatrixXd newCovars = MatrixXd::Zero(weightMean_.size(), weightMean_.size());
  return Trajectory(numWeights_, newMean, overlap_, newCovars, type_);;
}

void Trajectory::E_Step(const MatrixXd &mean_eStep, const MatrixXd &cov_eStep, Ref<VectorXd, 0, InnerStride<>> mean,
                        MatrixXd &cov) {
  cov = (cov_eStep + weightCovars_.inverse()).inverse();
  mean = (cov * (mean_eStep + (weightCovars_.inverse() * weightMean_))).col(0);
}

void Trajectory::M_Step(const MatrixXd &mean, const std::vector<MatrixXd> &cov) {
  weightMean_ = mean.colwise().mean().row(0);
  MatrixXd centered = mean.rowwise() - mean.colwise().mean();
  weightCovars_ = centered.transpose() * centered;

  for (int i = 0; i < mean.rows(); i++) {
    weightCovars_ += cov[i];
  }

  weightCovars_ /= mean.rows();
}

void Trajectory::M_Step2(const MatrixXd &mean, const std::vector<MatrixXd> &cov, const std::vector<MatrixXd> &RR,
                         const std::vector<MatrixXd> &RH, const std::vector<MatrixXd> &HH, const int sampleCount) {
  standardDev_ = 0;
  for (int i = 0; i < mean.rows(); i++) {
    standardDev_ += (HH[i] * cov[i]).trace();
    standardDev_ += RR[i](0, 0);
    standardDev_ -= 2 * (RH[i].transpose() * mean.row(i).transpose())(0, 0);
    standardDev_ += (mean.row(i) * HH[i] * mean.row(i).transpose())(0, 0);
  }

  standardDev_ /= mean.norm() * mean.rows() * numDim_ * sampleCount + 2;
}

void Trajectory::setBF() {
  if (type_ == Stroke) {
    basisFunctions_ = std::shared_ptr<BasisFunctions>(new StrokeBasisFunctions(numWeights_, overlap_));
  } else {
    basisFunctions_ = std::shared_ptr<BasisFunctions>(new PeriodicBasisFunctions(numWeights_, overlap_));
  }
}

void Trajectory::beginTimeMeasure() const{
  start = std::chrono::high_resolution_clock::now();
};

double Trajectory::endTimeMeasure() const {
  return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
};

CombinedTrajectory::CombinedTrajectory(const CombinedTrajectoryData &data) {
  for (int i = 0; i < static_cast<int>(data.activations_.size()); i++) {
    TrajectoryData trajectoryData(data.numBF_, data.numDim_, data.isStroke_, data.overlap_);
    trajectoryData.mean_ = std::vector<double>(data.means_[i]);
    trajectoryData.covariance_ = std::vector<double>(data.covariances_[i]);
    trajectories_.push_back(Trajectory(trajectoryData));
    activations_.push_back(
            constMatrix(data.activations_[i].data(), data.activations_[i].size() / trajectories_[i].numFunc_,
                        trajectories_[i].numFunc_)); // check row style
  }
}


CombinedTrajectory::CombinedTrajectory(const std::vector<Trajectory> &trajectories,
                                       const std::vector<MatrixXd> &activations) :
        trajectories_(trajectories), activations_(activations) {
  assert(trajectories.size() > 0);
  assert(trajectories.size() == activations.size());

  for (int i = 0; i < static_cast<int>(trajectories.size()); i++) {
    assert(trajectories.front().numDim_ == trajectories[i].numDim_);
    assert(activations[i].cols() == trajectories[i].numFunc_);
  }
}

double CombinedTrajectory::getActivation(const double time, const int index) const {
  int idx = 0;

  for (int i = 0; i < activations_[index].rows(); i++) {
    if (activations_[index](i, 0) > time) {
      idx = i - 1;
      break;
    } else if (i == activations_[index].rows() - 1) {
      idx = activations_[index].rows() - 2;
    }
  }

  if (activations_[index](idx, 0) == time) {
    return activations_[index](idx, 1);
  }

  double dist_A = std::fabs(time - activations_[index](idx, 0));
  double dist_B = std::fabs(time - activations_[index](idx + 1, 0));
  double ratio = 1 - (dist_A / (dist_A + dist_B));
  return (ratio * activations_[index](idx, 1)) + ((1 - ratio) * activations_[index](idx + 1, 1));
}

MatrixXd CombinedTrajectory::getValueCovars(const double time) const {
  MatrixXd out = MatrixXd::Zero(trajectories_.front().numDim_ * 2, trajectories_.front().numDim_ * 2);

  for (int i = 0; i < static_cast<int>(trajectories_.size()); i++) {
    const Trajectory &t = trajectories_[i];
    out += (t.getValueCovars(time) / getActivation(time, i)).inverse();
  }
  return out.inverse();
}


VectorXd CombinedTrajectory::getValueMean(const double time) const {
  VectorXd out = VectorXd::Zero(trajectories_.front().numDim_ * 2);

  for (int i = 0; i < static_cast<int>(trajectories_.size()); i++) {
    const Trajectory &t = trajectories_[i];
    out += (t.getValueCovars(time) / getActivation(time, i)).inverse() * t.getValueMean(time);
  }
  out = getValueCovars(time) * out;
  return out;
}


MatrixXd CombinedTrajectory::getValueMean(const VectorXd &time) const {
  MatrixXd out(trajectories_.front().numDim_ * 2, time.size());
  for (int i = 0; i < static_cast<int>(time.size()); i++) {
    out.col(i) = getValueMean(time(i));
  }
  return out;
}

std::vector<MatrixXd> CombinedTrajectory::getValueCovars(const VectorXd &time) const {
  std::vector<MatrixXd> out;
  for (int i = 0; i < static_cast<int>(time.size()); i++) {
    out.push_back(getValueCovars(time(i)));
  }
  return out;
}