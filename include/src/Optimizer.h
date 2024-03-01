/**
 * \file Optimizer.h
 * \author Malte Langosz
 * \brief A interface for parameter optimization algorithms.
 *
 * Version 0.1
 */

#ifndef __BL_OPTIMIZER_H__
#define __BL_OPTIMIZER_H__

#ifdef _PRINT_HEADER_
  #warning "Optimizer.h"
#endif

#include <string>
#include <vector>

#include <lib_manager/LibInterface.hpp>

namespace bolero {

  /**
   * @class Optimizer
   * Common interface for optimizers.
   */
  class Optimizer : public lib_manager::LibInterface {

  public:
    /**
     * Create an optimizer.
     * \param theManager libManager instance
     * \param libName name of the library
     * \param libVersion version of the library
     */
    Optimizer(lib_manager::LibManager *theManager,
              std::string libName, int libVersion)
      : lib_manager::LibInterface(theManager),
        libName(libName),
        libVersion(libVersion) {
    }

    virtual ~Optimizer() {}

    // LibInterface methods

    int getLibVersion() const {return libVersion;}
    const std::string getLibName() const {return libName;}
    virtual void createModuleInfo() {}

    // Optimizer methods

    /**
     * Initialize optimizer.
     * \param dimension dimension of parameter vector
     * \param config YAML-based configuration the optimizer, can be empty
     */
    virtual void init(int dimension, std::string config) = 0;
    /**
     * Get next individual/parameter vector for evaluation.
     * \param[out] p parameter vector, will be modified
     * \param numP expected number of parameters
     */
    virtual void getNextParameters(double *p, int numP) = 0;

    /**
     * Get best individual/parameter vector so far.
     * \param[out] p parameter vector, will be modified
     * \param numP expected number of parameters
     */
    virtual void getBestParameters(double *p, int numP) = 0;

    /**
     * Set feedbacks for the parameter vector.
     * \param feedbacks feedbacks for each step or for the episode, depends on
     *        the problem
     * \param numFeedbacks number of feedbacks
     */
    virtual void setEvaluationFeedback(const double *feedbacks,
                                       int numFeedbacks) = 0;

    /**
     * Check if the optimizer converged.
     * \return did the optimizer converge?
     */
    virtual bool isBehaviorLearningDone() const = 0;

    virtual void getNextParameterSet(double *p, int numP, int batchSize) const = 0;
    virtual void setParameterSetFeedback(const double *feedback, int numFeedbacksPerSet, int batchSize) = 0;
    virtual int getBatchSize() const = 0;

  protected:
    int dimension;
    std::string libName;
    int libVersion;
  }; // end of class definition Optimizer

} // end of namespace bolero

#endif // __BL_OPTIMIZER_H__
