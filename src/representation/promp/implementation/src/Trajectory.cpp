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
                       const int iterationLimit, const TrajectoryType type)
        : numWeights_(numWeights), numDim_(values.front().cols()), overlap_(overlap), type_(type) {
  setBF();
  imitate(timestamps, values,iterationLimit);
}

void Trajectory::imitate(const std::vector<VectorXd> &timestamps, const std::vector<MatrixXd> &values, const int iterationLimit) {
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
      val[i].block(timestamps[i].size() * j, 0, timestamps[i].size(), 1) = values[i].block(0, j, timestamps[i].size(),1);
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

  std::vector<MatrixXd> mean_eStep(values.size());

  for (size_t i = 0; i < values.size(); i++) {
    mean_eStep[i] = (BH[i] * R[i]);
  }

  std::vector<MatrixXd> cov_eStep(values.size());

  for (size_t i = 0; i < values.size(); i++) {
    cov_eStep[i] = (BH[i] * BH[i].transpose());
  }

  int sampleCount = 0;
  for (size_t i = 0; i < values.size(); i++) {
    sampleCount += values[i].rows();
  }

  int counter = 0;
  VectorXd weightMean_old;
  MatrixXd weightCovars_old;
  do {
    counter++;
    weightMean_old = weightMean_;
    weightCovars_old = weightCovars_;

    for (size_t i = 0; i < values.size(); i++) {
      E_Step(1. / standardDev_ * mean_eStep[i], 1. / standardDev_ * cov_eStep[i], means.row(i), covs[i]);
    }

    M_Step(means, covs, RR, mean_eStep, cov_eStep, sampleCount);
  } while (counter < iterationLimit && !weightMean_old.isApprox( weightMean_) && !weightCovars_old.isApprox(weightCovars_));

}


MatrixXd Trajectory::getWeightMean() const {
  return weightMean_;
};

MatrixXd Trajectory::getWeightCovars() const {
  return weightCovars_;
};

MatrixXd Trajectory::getValueMean(const VectorXd &time) const {
  MatrixXd out(numDim_ *numFunc_, time.size());
  
  MatrixXd bf = basisFunctions_->getValue(time);
  MatrixXd bfd = basisFunctions_->getValueDeriv(time);
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
      
      out.transpose().row((numFunc_ * y*numDim_*numFunc_)+ (numFunc_ * x)) = (value *
                                             weightCovars_.block(y * numWeights_, x * numWeights_, numWeights_,
                                                                 numWeights_) *
                                             value.transpose()).diagonal();
      
      out.transpose().row((numFunc_ * y*numDim_*numFunc_)+ (numFunc_ * x) + 1) = (value *
                                                 weightCovars_.block(y * numWeights_, x * numWeights_, numWeights_,
                                                                     numWeights_) *
                                                 valueDeriv.transpose()).diagonal();
      
      out.transpose().row((numFunc_ * y*numDim_*numFunc_) + (numFunc_ * x)+(numDim_*numFunc_)) = (valueDeriv *
                                                 weightCovars_.block(y * numWeights_, x * numWeights_, numWeights_,
                                                                     numWeights_) *
                                                 value.transpose()).diagonal();

      out.transpose().row((numFunc_ * y*numDim_*numFunc_) + (numFunc_ * x) + (numDim_*numFunc_) +1) = (valueDeriv *
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

  for (unsigned i = 0; i < conditionPoints_.size(); i++) {
    const ConditionPoint &point = conditionPoints_[i];
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

Trajectory Trajectory::sampleTrajectoty(unsigned& seed) const {
  std::default_random_engine generator(seed);
  MatrixXd A = weightCovars_.selfadjointView<Lower>().llt().matrixL();
  VectorXd z(weightMean_.size());
  
  std::normal_distribution<double> normalDist(0, 1);

  for (int i = 0; i < weightMean_.size(); i++) {
    z(i) = normalDist(generator);
  }

  VectorXd newMean = weightMean_ + (A * z);
  MatrixXd newCovars = MatrixXd::Zero(weightMean_.size(), weightMean_.size());
  seed = generator();
  return Trajectory(numWeights_, newMean, overlap_, newCovars, type_);;


}
Trajectory Trajectory::sampleTrajectoty() const {
  unsigned seed = time(0);
  return sampleTrajectoty(seed);
}

void Trajectory::E_Step(const MatrixXd &mean_eStep, const MatrixXd &cov_eStep, Ref<VectorXd, 0, InnerStride<>> mean,
                        MatrixXd &cov) {
  cov = (cov_eStep + weightCovars_.inverse()).inverse();
  mean = (cov * (mean_eStep + (weightCovars_.inverse() * weightMean_))).col(0);
}

void Trajectory::M_Step(const MatrixXd &mean, const std::vector<MatrixXd> &cov, const std::vector<MatrixXd> &RR,
                         const std::vector<MatrixXd> &RH, const std::vector<MatrixXd> &HH, const int sampleCount) {

  weightMean_ = mean.colwise().mean().row(0);
  MatrixXd centered = mean.rowwise() - mean.colwise().mean();
  weightCovars_ = centered.transpose() * centered;

  for (int i = 0; i < mean.rows(); i++) {
    weightCovars_ += cov[i];
  }

  weightCovars_ /= mean.rows();                           

  standardDev_ = 0;
  for (int i = 0; i < mean.rows(); i++) {
    standardDev_ += (HH[i] * cov[i]).trace();
    standardDev_ += RR[i](0, 0);
    standardDev_ -= 2 * (RH[i].transpose() * mean.row(i).transpose())(0, 0);
    standardDev_ += (mean.row(i) * HH[i] * mean.row(i).transpose())(0, 0);
  }

  standardDev_ /= mean.norm() * mean.rows() * numDim_ * sampleCount + 2; // magic number from the paper
}

void Trajectory::setBF() {
  if (type_ == Stroke) {
    basisFunctions_ = std::shared_ptr<BasisFunctions>(new StrokeBasisFunctions(numWeights_, overlap_));
  } else {
    basisFunctions_ = std::shared_ptr<BasisFunctions>(new PeriodicBasisFunctions(numWeights_, overlap_));
  }
}
