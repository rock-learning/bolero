from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool


cdef extern from "lib_manager/LibManager.hpp" namespace "lib_manager":
  cdef cppclass LibManager
  cdef cppclass LibInterface


cdef extern from "Behavior.h" namespace "bolero":
  cdef cppclass Behavior:
    Behavior() except +
    void init(int numInputs, int numOutputs) except +
    void setInputs(double *values, int numInputs) except +
    void getOutputs(double *values, int numOutputs) except +
    int getNumInputs() except +
    int getNumOutputs() except +
    void step() except +
    bool canStep() except +

cdef extern from "LoadableBehavior.h" namespace "bolero":
  cdef cppclass LoadableBehavior(Behavior):
    bool initialize(string &initialConfigPath) except +
    bool configure(string &configPath) except +
    bool configureYaml(string &yaml) except +


cdef extern from "BehaviorSearch.h" namespace "bolero":
  cdef cppclass BehaviorSearch:
    BehaviorSearch(LibManager *theManager, string &libName,
                   int libVersion) except +
    int getLibVersion() except +
    string getLibName() except +
    void createModuleInfo() except +
    void init(int numInputs, int numOutputs) except +
    Behavior* getNextBehavior() except +
    Behavior* getBestBehavior() except +
    void setEvaluationFeedback(double *feedbacks, int numFeedbacks) except +
    void writeResults(string &resultPath) except +
    Behavior* getBehaviorFromResults(string &resultPath) except +
    bool isBehaviorLearningDone() except +

cdef extern from "Environment.h" namespace "bolero":
  cdef cppclass Environment:
    Environment(LibManager *theManager, string &libName,
                int libVersion) except +
    int getLibVersion() except +
    string getLibName() except +
    void createModuleInfo() except +
    void init() except +
    void reset() except +
    int getNumInputs() except +
    int getNumOutputs() except +
    void getOutputs(double *values, int numOutputs) except +
    void setInputs(double *values, int numInputs) except +
    void stepAction() except +
    void setTestMode(bool b) except +
    bool isEvaluationDone() except +
    int getFeedback(double *feedback) except +
    bool isBehaviorLearningDone() except +
    bool isContextual() except +

cdef extern from "ContextualEnvironment.h" namespace "bolero":
  cdef cppclass ContextualEnvironment(Environment):
    double* request_context(double *context, int numContext) except +
    int get_num_context_dims() except +

cdef extern from "ParameterizedEnvironment.h" namespace "bolero":
  cdef cppclass ParameterizedEnvironment(Environment):
    int getNumParameters()
    void getParameters(double *values, int numParameters)
    void setParameters(const double *values, int numParameters)

cdef extern from "Optimizer.h" namespace "bolero":
  cdef cppclass Optimizer:
    Optimizer(LibManager *theManager, libName, int libVersion) except +
    int getLibVersion() except +
    string getLibName() except +
    void createModuleInfo() except +
    void init(int dimension) except +
    void getNextParameters(double *p, int numP) except +
    void getBestParameters(double *p, int numP) except +
    void setEvaluationFeedback(double *feedbacks, int numFeedbacks) except +
    bool isBehaviorLearningDone() except +
    vector[double*] getNextParameterSet() except +
    void setParameterSetFeedback(vector[double] feedback) except +


cdef extern from "BLLoader.h" namespace "bolero::bl_loader":
  cdef cppclass BLLoader:
    BLLoader() except +
    int getLibVersion() except +
    string getLibName() except +
    void addPythonModulePath(string &path) except +
    void loadConfigFile(string &config_file) except +
    void loadLibrary(string &libPath, void *config) except +
    void unloadLibrary(string &name) except +
    void addLibrary(LibInterface *lib) except +
    Optimizer* acquireOptimizer(string &name) except +
    BehaviorSearch* acquireBehaviorSearch(string &name) except +
    Environment* acquireEnvironment(string &name) except +
    LoadableBehavior* acquireBehavior(string &name) except +
    ContextualEnvironment* acquireContextualEnvironment(string &name) except +
    ParameterizedEnvironment* acquireParameterizedEnvironment(string &name) except +
    void releaseLibrary(string &name) except +
    void dumpTo(string &file) except +
