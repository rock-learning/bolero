#pragma once
#include<vector>

namespace promp {
  struct TrajectoryData {
    TrajectoryData(const int numBF, const int numDim, const bool isStroke,const double overlap);

    std::vector<double> mean_;
    std::vector<double> covariance_;
    std::vector<double> conditions_;
    int numBF_;
    int numDim_;
    bool isStroke_;
    double overlap_;
    void sampleTrajectoryData(TrajectoryData &traj) const;

    void stepCov(const double timestamp, double *covs, int numCovs) const;

    void step(const double timestamp, double *values, int numValues) const;

    void imitate(const double *sizes, const int numSizes, const double *timestamps, const int numTimestamps, const double *values,
                 const int numValues);

    void getValues(const double *timestamps, const int numTimestamps, double *means, int numMeans, double *covars,
                   int numCovars) const;

    void condition(const int count, const double *points, const int numPoints);
  };

  struct CombinedTrajectoryData {
    std::vector<std::vector<double> > means_;
    std::vector<std::vector<double> > covariances_;
    std::vector<std::vector<double> > activations_;
    int numBF_;
    int numDim_;
    bool isStroke_;
    double overlap_;
    void addTrajectory(const TrajectoryData trajectory, const double *activation, const int numActivation);

    void step(const double timestamp, double *values, int numValues, double *covs, int numCovs) const;

    void getValues(const double *timestamps, const int numTimestamps, double *means, int numMeans, double *covars,
                   int numCovars) const;
  };


/*
  void getTrajectory(int dimensions, double* weights, int numWeights, double* times, int numTimes, double* out, int numOut);  
  void getWeights(double* demonstrations, int numDemonstrations, double* times, int numTimes, double* weights, int numWeights);
  void getStatistics(int dimensions, double* weights, int numWeights, double* means, int numMeans, double* covariances, int numCovariances);
  void getStatistics2(int numWeights, int num, double* times,int numTimes, double* values, int numValues, double* means, int numMeans, double* covariances, int numCovariances);
  void getTrajectory2(int dimensions, double* weights, int numWeights, double* covariances, int numCovariances,double* times, int numTimes, double* values, int numvalues,double* confi, int numConfi );
  void getStatistics3(int num, double* times,int numTimes, double* values, int numValues, TrajectoryData& simple);
  void getTrajectory3(double* times, int numTimes, double* values, int numvalues,double* confi, int numConfi, TrajectoryData& simple);*/
}
