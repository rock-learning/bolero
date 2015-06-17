#include "PyOptimizer.h"
#include "Helper.h"
#include <cassert>


namespace bolero { namespace bl_loader {

  PyOptimizer::PyOptimizer(lib_manager::LibManager *theManager,
                           std::string libName, int libVersion)
    : bolero::Optimizer(theManager, libName, libVersion),
      optimizer(0),
      dimension(0)
  {
    optimizer = PythonInterpreter::instance()
        .import("bolero.utils.module_loader")
        .function("optimizer_from_yaml").call()
        .returnObject();
  }

  void PyOptimizer::init(int dimension) {
    this->dimension = dimension;
    optimizer.method("init").pass(INT).call(dimension);
  }

  void PyOptimizer::getNextParameters(double *p, int numP) {
    // TODO direct array access
    optimizer.method("get_next_parameters").pass(ONEDARRAY).call(array);
  }

  void PyOptimizer::getBestParameters(double *p, int numP) {
    // TODO direct array access
    optimizer.method("get_best_parameters").pass(ONEDARRAY).call(array);
  }

  void PyOptimizer::setEvaluationFeedback(const double *feedbacks,
                                          int numFeedbacks) {
    // TODO direct array access
    optimizer.method("set_evaluation_feedback").pass(ONEDARRAY).call(array);
  }

  bool PyOptimizer::isBehaviorLearningDone() const {
    return optimizer.method("is_behavior_learning_done").call().returnObject().asBool();
  }

  std::vector<double*> PyOptimizer::getNextParameterSet() const {
    throw std::runtime_error("PyOptimizer::getNextParameterSet() not implemented yet!");
  }

  void PyOptimizer::setParameterSetFeedback(const std::vector<double> feedback) {
	throw std::runtime_error("PyOptimizer::setParameterSetFeedback() not implemented yet!");
  }


}}
