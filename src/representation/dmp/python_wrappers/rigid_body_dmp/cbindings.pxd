# Declare class that will be wrapped by Cython
from libcpp cimport bool
from libcpp.string cimport string
  	
cdef extern from "lib_manager/LibInterface.hpp" namespace "lib_manager":
  cdef cppclass LibManager

cdef extern from "CanonicalSystem.h" namespace "dmp::CanonicalSystem":
    double calculateAlpha(double lastPhaseValue, double dt, double executionTime)
    double calculateAlpha(double lastPhaseValue, int numPhases)
	

cdef extern from "RbfFunctionApproximator.h" namespace "dmp::RbfFunctionApproximator":
    void calculateCenters(double lastPhaseValue, double executionTime,
                          double dt, unsigned numCenters,
                          double overlap, double* centers,
                          double* widths)

cdef extern from "RigidBodyDmp.h" namespace "dmp::RigidBodyDmp":
    void determineForces(double* positions, int positionRows,
                         int positionCols, double* rotations,
                         int rotationRows, int rotationCols,
                         double* forces, int forcesRows, int forcesCols,
                         double executionTime, double dt,
                         double alphaZ, double betaZ)

    void determineForces(double* positions, int positionRows,
                         int positionCols,double* forces, int forcesRows,
                         int forcesCols, double executionTime, double dt,
                         double alphaZ, double betaZ)

cdef extern from "RigidBodyDmp.h" namespace "dmp":
    cdef cppclass RigidBodyDmp:    
        RigidBodyDmp(LibManager* manager)
        void setInputs(double *values, int numInputs)
        void getOutputs(double *values, int numOutputs)
        void step()
        bool canStep()
        bool configureYaml(string& yaml)
        bool initializeYaml(string& yaml)
        void getActivations(double s, bool normalized, double* activations, int size)
        void setWeights(double *weights, int rows, int cols)
        void getPhases(double* phases, int len)

