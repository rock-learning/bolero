/**
 * \file Environment.h
 * \author Malte Langosz
 * \brief A interface for a environment for the machine learning framework.
 *
 * Version 0.1
 */

#ifndef BL_FRAMEWORK_ENVIRONMENT_H
#define BL_FRAMEWORK_ENVIRONMENT_H

#ifdef _PRINT_HEADER_
  #warning "Environment.h"
#endif

#include <lib_manager/LibInterface.hpp>
#include <string>

namespace bolero {

  class Behavior;

  /**
   * @class Environment
   * Common interface for environments.
   * An environment can execute actions, measure states and compute rewards.
   * It defines a learning problem.
   */
  class Environment : public lib_manager::LibInterface {

  public:
    /**
     * Create an environment.
     * \param theManager libManager instance
     * \param libName name of the library
     * \param libVersion version of the library
     */
    Environment(lib_manager::LibManager *theManager,
                const std::string &libName, int libVersion) :
      lib_manager::LibInterface(theManager), libName(libName),
      libVersion(libVersion) {
    }

    virtual ~Environment() {}

    // LibInterface methods

    virtual int getLibVersion() const {return libVersion;}
    virtual const std::string getLibName() const {return libName;}
    virtual void createModuleInfo() {}

    // Environment methods

    /**
     * Initialize environment.
     * \param config YAML-based configuration the environment, can be empty
     */
    virtual void init(std::string config) = 0;

    /**
     * Reset state of the environment.
     */
    virtual void reset() = 0;

    virtual bool isContextual(){return false;}

    /**
     * Get number of environment inputs.
     * \return number of inputs
     */
    virtual int getNumInputs() const = 0;

    /**
     * Get number of environment outputs.
     * \return Number of environment outputs
     */
    virtual int getNumOutputs() const = 0;

    /**
     * Get outputs.
     * \param values outputs, e.g. current state of the system
     * \param numOutputs expected number of outputs
     */
    virtual void getOutputs(double *values, int numOutputs) const = 0;

    /**
     * Set input for the next step.
     * \param[out] values inputs e.g. desired state of the system
     * \param numInputs number of inputs
     */
    virtual void setInputs(const double *values, int numInputs) = 0;

    /**
     * Take a step in the environment.
     */
    virtual void stepAction() = 0;

    /**
     * Sets whether the environment is in training or test mode.
     * \param test test mode?
     */
    virtual void setTestMode(bool test) {}

    /**
     * Is the evaluation of the behavior finished?
     * \return is the evaluation finished?
     */
    virtual bool isEvaluationDone() const = 0;

    /**
     * Is the evaluation of the behavior aborted?
     * \return is the evaluation aborted?
     */
    virtual bool isEvaluationAborted() {return false;}

    /**
     * Get feedbacks from the last episode.
     * \param[out] feedback array, will be filled with feedbacks
     * \return how many rewards were assigned for the whole evaluation
     */
    virtual int getFeedback(double *feedback) const = 0;

    /**
     * Get feedbacks from the last step.
     * \param[out] feedback array, will be filled with feedbacks
     * \return how many rewards were assigned for the whole evaluation
     */
      virtual int getStepFeedback(double *feedback) const {*feedback=0; return 0;}

    /**
     * Check if the behavior learning is finished.
     * \return is the learning of a behavior finished?
     */
    virtual bool isBehaviorLearningDone() const = 0;

  protected:
    std::string libName;
    int libVersion;

  }; // end of class definition Environment

} // end of namespace bolero

#endif // BL_FRAMEWORK_ENVIRONMENT_H
