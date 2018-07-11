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

    /**
     * Calculates the mean values (e.g. angles/coordinates) and their derivatives for a given vector of timestamps
     */
    MatrixXd getValueMean(const VectorXd &time) const;

    /**
     * Calculates the mean values (e.g. angles/coordinates) and their derivatives for a given timestamp
     */
    VectorXd getValueMean(const double time) const;

    /**
     * Calculates the covariance of the values (e.g. angles/coordinates) and their derivatives for a given vector of timestamps
     */
    MatrixXd getValueCovars(const VectorXd &time) const;

    /**
     * Calculates the covariance of the values (e.g. angles/coordinates) and their derivatives for a given timestamp
     */
    VectorXd getValueCovars(const double time) const;

    inline void setConditions(const std::vector<ConditionPoint>& conditionPoints){ conditionPoints_ = conditionPoints; };

    void getData(TrajectoryData &data) const;

    /**
     * Samples a Trajectory from the current distribution with given seed
     * seed is only used through this very run. however its adjusted to keep track of it 
     */ 
    Trajectory sampleTrajectoty(unsigned& seed) const;

    /**
     * Samples a Trajectory from the current distribution with random seed
     */ 
    Trajectory sampleTrajectoty() const;

    const int numWeights_; 
    const int numDim_; // num of dim per function (pos/acc)
    const double overlap_;
    static constexpr int numFunc_ = 2; // position and acceleration

  private:
    /**
     * Runs the E-Step from the ProMBayesian Multi-Task Reinforcement Learning Paper
     * see https://hal.inria.fr/inria-00475214/document
     */
    void E_Step(const MatrixXd &mean_eStep, const MatrixXd &cov_eStep, Ref<VectorXd, 0, InnerStride<>> mean, MatrixXd &cov);
    
    /**
     * Runs the M-Step from the ProMBayesian Multi-Task Reinforcement Learning Paper
     * see https://hal.inria.fr/inria-00475214/document
     */
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