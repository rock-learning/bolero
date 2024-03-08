#include "PyBehaviorSearch.h"
#include "PyBehavior.h"
#include <cassert>
#include <stdexcept>
#include <configmaps/ConfigMap.hpp>


namespace bolero {
  namespace bl_loader {

    PyBehaviorSearch::PyBehaviorSearch(lib_manager::LibManager *theManager,
                                       const std::string libName, int libVersion)
      : bolero::BehaviorSearch(theManager, libName, libVersion), behaviorSearch(0),
        behavior(0), bestBehavior(0)
    {
    }

    PyBehaviorSearch::~PyBehaviorSearch()
    {
      if(behavior)
        delete behavior;
      if(bestBehavior)
        delete bestBehavior;
    }

    void PyBehaviorSearch::init(int numInputs, int numOutputs, std::string config) {
      pthread_mutex_lock (&PythonInterpreter::mutex);
      if(config == "")
        config = "BehaviorSearch:\n    type: " + libName;
      behaviorSearch = PythonInterpreter::instance()
        .import("bolero.utils.module_loader")
        ->function("behavior_search_from_yaml_string").pass(STRING).call(&config)
        .returnObject();
      if(!behaviorSearch) {
        std::runtime_error("Behavior search construction failed");
      }
      behaviorSearch->method("init")
        .pass(INT).pass(INT).call(numInputs, numOutputs);
      pthread_mutex_unlock (&PythonInterpreter::mutex);
    }

    bolero::Behavior* PyBehaviorSearch::getNextBehavior() {
      pthread_mutex_lock (&PythonInterpreter::mutex);
      if(behavior)
        delete behavior;
      shared_ptr<Object> behaviorObject = behaviorSearch
        ->method("get_next_behavior").call().returnObject();
      behavior = PyBehavior::fromPyObject(behaviorObject);
      pthread_mutex_unlock (&PythonInterpreter::mutex);
      return behavior;
    }

    bolero::Behavior* PyBehaviorSearch::getBestBehavior() {
      pthread_mutex_lock (&PythonInterpreter::mutex);
      if(bestBehavior)
        delete bestBehavior;
      shared_ptr<Object> behaviorObject = behaviorSearch
        ->method("get_best_behavior").call().returnObject();
      bestBehavior = PyBehavior::fromPyObject(behaviorObject);
      pthread_mutex_unlock (&PythonInterpreter::mutex);
      return bestBehavior;
    }

    void PyBehaviorSearch::setEvaluationFeedback(
                                                 const double *feedbacks, int numFeedbacks) {
      pthread_mutex_lock (&PythonInterpreter::mutex);
      behaviorSearch->method("set_evaluation_feedback")
        .pass(ONEDCARRAY).call(feedbacks, numFeedbacks);
      pthread_mutex_unlock (&PythonInterpreter::mutex);
    }

    void PyBehaviorSearch::setStepFeedback(
                                           const double *feedbacks, int numFeedbacks) {
      pthread_mutex_lock (&PythonInterpreter::mutex);
      behaviorSearch->method("set_step_feedback")
        .pass(ONEDCARRAY).call(feedbacks, numFeedbacks);
      pthread_mutex_unlock (&PythonInterpreter::mutex);
    }

    void PyBehaviorSearch::setEvaluationDone(bool aborted) {
      pthread_mutex_lock (&PythonInterpreter::mutex);
      behaviorSearch->method("set_evaluation_done")
        .pass(BOOL).call(aborted);
      pthread_mutex_unlock (&PythonInterpreter::mutex);
    }

    void PyBehaviorSearch::writeResults(const std::string &resultPath) {
      pthread_mutex_lock (&PythonInterpreter::mutex);
      std::string path = resultPath;
      behaviorSearch->method("write_results").pass(STRING).call(&path);
      pthread_mutex_unlock (&PythonInterpreter::mutex);
    }

    bolero::Behavior* PyBehaviorSearch::getBehaviorFromResults(
                                                               const std::string &resultPath) {
      pthread_mutex_lock (&PythonInterpreter::mutex);
      if(behavior)
        delete behavior;
      std::string path = resultPath;
      shared_ptr<Object> behaviorObject = behaviorSearch
        ->method("get_behavior_from_results").pass(STRING).call(&path)
        .returnObject();
      behavior = PyBehavior::fromPyObject(behaviorObject);
      pthread_mutex_unlock (&PythonInterpreter::mutex);
      return behavior;
    }

    bool PyBehaviorSearch::isBehaviorLearningDone() const {
      pthread_mutex_lock (&PythonInterpreter::mutex);
      bool v = behaviorSearch->method("is_behavior_learning_done").call()
        .returnObject()->asBool();
      pthread_mutex_unlock (&PythonInterpreter::mutex);
      return v;
    }

    std::string PyBehaviorSearch::getBehaviorBatch() const {
      pthread_mutex_lock (&PythonInterpreter::mutex);
      shared_ptr<Object> pyObject = behaviorSearch
        ->method("get_behavior_batch").call()
        .returnObject();
      std::string r = pyObject->asString();
      pthread_mutex_unlock (&PythonInterpreter::mutex);
      return r;
    }

    void PyBehaviorSearch::setBatchFeedback(const double* batchFeedback, int numFeedbacksPerBatch, int batchSize) {
      pthread_mutex_lock (&PythonInterpreter::mutex);
      behaviorSearch->method("set_batch_feedback")
        .pass(ONEDCARRAY).pass(INT).pass(INT).call(batchFeedback, batchSize*numFeedbacksPerBatch, numFeedbacksPerBatch, batchSize);
      pthread_mutex_unlock (&PythonInterpreter::mutex);
    }

    Behavior* PyBehaviorSearch::getBehaviorFromString(std::string &behaviorString) {
      pthread_mutex_lock (&PythonInterpreter::mutex);
      if(behavior)
        delete behavior;
      shared_ptr<Object> behaviorObject = behaviorSearch
        ->method("get_behavior_from_string").pass(STRING).call(&behaviorString)
        .returnObject();
      behavior = PyBehavior::fromPyObject(behaviorObject);
      pthread_mutex_unlock (&PythonInterpreter::mutex);
      return behavior;
    }

  }
}
