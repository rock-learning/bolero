#pragma once
#include<vector>

namespace promp {
  struct TrajectoryData {
    TrajectoryData(const int numBF, const int numDim, const bool isStroke,const double overlap);

    std::vector<double> mean_;
    std::vector<double> covariance_;
    std::vector<double> conditions_;
    int iterationLimit_;
    int numBF_;
    int numDim_;
    bool isStroke_;
    double overlap_;
    unsigned randomState_;

    void sampleTrajectoryData(TrajectoryData &traj);

    void stepCov(const double timestamp, double *covs, int numCovs) const;

    void step(const double timestamp, double *values, int numValues) const;

    void imitate(const double *sizes, const int numSizes, const double *timestamps, const int numTimestamps, const double *values,
                 const int numValues);

    void getValues(const double *timestamps, const int numTimestamps, double *means, int numMeans, double *covars,
                   int numCovars) const;

    void condition(const int count, const double *points, const int numPoints);
  };
}
