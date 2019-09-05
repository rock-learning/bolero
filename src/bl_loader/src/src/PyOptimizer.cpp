#include "PyOptimizer.h"
#include <cassert>
#include <stdexcept>
#include <algorithm>


namespace bolero {
  namespace bl_loader {

    PyOptimizer::PyOptimizer(lib_manager::LibManager *theManager,
                             std::string libName, int libVersion)
      : bolero::Optimizer(theManager, libName, libVersion),
        optimizer(0), dimension(0)
    {
    }

    void PyOptimizer::init(int dimension, std::string config) {
      if(config == "")
        config = "Optimizer:\n    type: " + libName;
      optimizer = PythonInterpreter::instance()
        .import("bolero.utils.module_loader")
        ->function("optimizer_from_yaml_string").pass(STRING).call(&config)
        .returnObject();
      if(!optimizer)
        std::runtime_error("Optimizer construction failed");
      this->dimension = dimension;
      optimizer->method("init").pass(INT).call(dimension);
    }

    void PyOptimizer::getNextParameters(double *p, int numP) {
      optimizer->method("get_next_parameters").pass(ONEDCARRAY).call(p, numP);
    }

    void PyOptimizer::getBestParameters(double *p, int numP) {
      shared_ptr<Object> result = optimizer->method("get_best_parameters")
        .call().returnObject();
      shared_ptr<std::vector<double> > paramsVector = result->as1dArray();

      const int numParams = (int) paramsVector->size();
      if(numParams != numP)
        throw std::runtime_error("Expected another number of parameters");
      std::copy(paramsVector->begin(), paramsVector->end(), p);
    }

    void PyOptimizer::setEvaluationFeedback(const double *feedbacks,
                                            int numFeedbacks) {
      optimizer->method("set_evaluation_feedback").pass(ONEDCARRAY).call(
                                                                         feedbacks, numFeedbacks);
    }

    bool PyOptimizer::isBehaviorLearningDone() const {
      return optimizer->method("is_behavior_learning_done")
        .call().returnObject()->asBool();
    }

    void PyOptimizer::getNextParameterSet(double *p, int numP, int batchSize) const {
      optimizer->method("get_next_parameter_batch").pass(ONEDCARRAY).call(p, numP, batchSize);
    }

    void PyOptimizer::setParameterSetFeedback(const double *feedback, int numFeedbacksPerSet, int batchSize) {
      optimizer->method("set_batch_feedback").pass(ONEDCARRAY)
        .call(feedback, numFeedbacksPerSet, batchSize);
    }

    int PyOptimizer::getBatchSize() const {
      return optimizer->method("get_batch_size")
        .call().returnObject()->asInt();
    }

  }
}
