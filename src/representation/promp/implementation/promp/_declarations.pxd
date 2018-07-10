from libcpp cimport bool
from libcpp.vector cimport vector



cdef extern from "../src/promp.h" namespace "promp":
    cdef cppclass TrajectoryData:
        vector[double] mean_
        vector[double] covariance_
        vector[double] conditions_
        int iterationLimit_
        int numBF_
        int numDim_
        bool isStroke_
        double overlap_
        unsigned int randomState_
        TrajectoryData(int numBF, int numDim, bool isStroke, double overlap)
        void sampleTrajectoryData(TrajectoryData traj) except +
        void stepCov(double timestamp, double * covs, int numCovs) except +
        void step(double timestamp, double * values, int numValues) except +
        void imitate(double * sizes, int numSizes, double * timestamps, int numTimestamps, double * values, int numValues) except +
        void getValues(double * timestamps, int numTimestamps, double * means, int numMeans, double * covars, int numCovars) except +
        void condition(int count, double * points, int numPoints) except +
