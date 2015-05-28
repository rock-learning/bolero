from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool


cdef extern from "lib_manager/LibManager.hpp" namespace "lib_manager":
  cdef cppclass LibManager
  cdef cppclass LibInterface


cdef extern from "Behavior.h" namespace "bolero":
  cdef cppclass Behavior:
    Behavior(int numInputs, int numOutputs)
    void setInputs(double *values, int numInputs)
    void getOutputs(double *values, int numOutputs)
    int getNumInputs()
    int getNumOutputs()
    void step()
    bool canStep()

cdef extern from "LoadableBehavior.h" namespace "bolero":
  cdef cppclass LoadableBehavior(Behavior):
    bool initialize(string &initialConfigPath)
    bool configure(string &configPath)
    bool configureYaml(string &yaml)


cdef extern from "BehaviorSearch.h" namespace "bolero":
  cdef cppclass BehaviorSearch:
    BehaviorSearch(LibManager *theManager, string &libName,
                   int libVersion)
    int getLibVersion()
    string getLibName()
    void createModuleInfo()
    void init(int numInputs, int numOutputs)
    Behavior* getNextBehavior()
    Behavior* getBestBehavior()
    void setEvaluationFeedback(double *feedbacks,
                               int numFeedbacks)
    void writeResults(string &resultPath)
    Behavior* getBehaviorFromResults(string &resultPath)
    bool isBehaviorLearningDone()

cdef extern from "Environment.h" namespace "bolero":
  cdef cppclass Environment:
    Environment(LibManager *theManager, string &libName,
                int libVersion)
    int getLibVersion()
    string getLibName()
    void createModuleInfo()
    void init()
    void reset()
    int getNumInputs()
    int getNumOutputs()
    void getOutputs(double *values, int numOutputs)
    void setInputs(double *values, int numInputs)
    void stepAction()
    void setTestMode(bool b)
    bool isEvaluationDone()
    int getFeedback(double *feedback)
    bool isBehaviorLearningDone()
    bool isContextual()

cdef extern from "ContextualEnvironment.h" namespace "bolero":
  cdef cppclass ContextualEnvironment(Environment):
    double* request_context(double *context, int numContext)
    int get_num_context_dims()


cdef extern from "Optimizer.h" namespace "bolero":
  cdef cppclass Optimizer:
    Optimizer(LibManager *theManager, libName, int libVersion)
    int getLibVersion()
    string getLibName()
    void createModuleInfo()
    void init(int dimension)
    void getNextParameters(double *p, int numP)
    void getBestParameters(double *p, int numP)
    void setEvaluationFeedback(double *feedbacks, int numFeedbacks)
    bool isBehaviorLearningDone()
    vector[double*] getNextParameterSet()
    void setParameterSetFeedback(vector[double] feedback)


cdef extern from "BLLoader.h" namespace "bolero::bl_loader":
  cdef cppclass BLLoader:
    BLLoader()
    int getLibVersion()
    string getLibName()
    void addPythonModulePath(string &path)
    void loadConfigFile(string &config_file)
    void loadLibrary(string &libPath, void *config)
    void unloadLibrary(string &name)
    void addLibrary(LibInterface *lib)
    Optimizer* acquireOptimizer(string &name) except +
    BehaviorSearch* acquireBehaviorSearch(string &name) except +
    Environment* acquireEnvironment(string &name) except +
    LoadableBehavior* acquireBehavior(string &name) except +
    ContextualEnvironment* acquireContextualEnvironment(string &name) except +
    void releaseLibrary(string &name)
    void dumpTo(string &file)
