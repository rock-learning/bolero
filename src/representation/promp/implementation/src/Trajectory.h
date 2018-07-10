#pragma once

#include <Eigen/Dense>
#include "BasisFunctions.h"
#include <vector>
#include "promp.h"
#include <chrono>
#include <memory>

namespace promp {
  using namespace Eigen;

  enum TrajectoryType {
    Stroke,
    Periodic
  };

  struct ConditionPoint {
    double timestamp;
    int dimension;
    int derivative;
    double mean;
    double variance;

    static std::vector<ConditionPoint> fromMatrix(MatrixXd pointMatrix);
    const static unsigned NUM_FIELDS = 5;
  };

  class TrajectoryBase {
  public:
    virtual MatrixXd getValueMean(const VectorXd &time) const = 0;

    virtual VectorXd getValueMean(const double time) const = 0;

    virtual std::vector<MatrixXd> getValueCovars(const VectorXd &time) const =0;

    virtual MatrixXd getValueCovars(const double time) const =0;
  };


  class Trajectory{
  public:
    Trajectory(const TrajectoryData& data);

    Trajectory(const int numWeights, const VectorXd &weights, const double overlap, const MatrixXd &covars = MatrixXd::Zero(0, 0),
               const TrajectoryType type = Stroke);

    Trajectory(const std::vector<VectorXd> &timestamps, const std::vector<MatrixXd> &values, const double overlap, int numWeights = 30,
               const int iterationLimit = 100,const TrajectoryType type = Stroke);

    void imitate(const std::vector<VectorXd> &timestamps, const std::vector<MatrixXd> &values, const int imitationLimit = 100);

    MatrixXd getWeightMean() const;

    MatrixXd getWeightCovars() const;

    MatrixXd getValueMean(const VectorXd &time) const;

    VectorXd getValueMean(const double time) const;

    MatrixXd getValueCovars(const VectorXd &time) const;

    VectorXd getValueCovars(const double time) const;

    void setConditions(const std::vector<ConditionPoint>& conditionPoints){
      conditionPoints_ = conditionPoints;
    };

    void getData(TrajectoryData &data) const;

    Trajectory sampleTrajectoty(unsigned& seed) const;
    Trajectory sampleTrajectoty() const;

    const int numWeights_;
    const int numDim_;
    const double overlap_;
    static constexpr int numFunc_ = 2; // position and acceleration

  private:
    void E_Step(const MatrixXd &mean_eStep, const MatrixXd &cov_eStep, Ref<VectorXd, 0, InnerStride<>> mean, MatrixXd &cov);

    void M_Step(const MatrixXd &mean, const std::vector<MatrixXd> &cov, const std::vector<MatrixXd> &RR,
                 const std::vector<MatrixXd> &RH, const std::vector<MatrixXd> &HH, const int sampleCount);

    void setBF();

    void condition(VectorXd& weightMean,MatrixXd& weightCovars) const;

    VectorXd weightMean_;
    MatrixXd weightCovars_;
    
    std::vector<ConditionPoint> conditionPoints_;
    double standardDev_;
    TrajectoryType type_;
    std::shared_ptr<BasisFunctions> basisFunctions_ = nullptr;
    mutable std::chrono::high_resolution_clock::time_point start;
  };
}