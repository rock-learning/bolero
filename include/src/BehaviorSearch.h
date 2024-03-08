/**
 * \file BehaviorSearch.h
 * \author Malte Langosz
 * \brief A interface for a machine learning algorithm.
 *
 * Version 0.1
 */

#ifndef __BL_BEHAVIOR_SEARCH_H__
#define __BL_BEHAVIOR_SEARCH_H__

#ifdef _PRINT_HEADER_
  #warning "BehaviorSearch.h"
#endif

#include <string>

#include <lib_manager/LibInterface.hpp>

#include "Behavior.h"

namespace bolero {

  /**
   * @class BehaviorSearch
   * Common interface for behavior search.
   */
  class BehaviorSearch : public lib_manager::LibInterface {

  public:
    /**
     * Create a behavior search.
     * \param theManager libManager instance
     * \param libName name of the library
     * \param libVersion version of the library
     */
    BehaviorSearch(lib_manager::LibManager *theManager,
                   const std::string &libName,
                   int libVersion)
        : lib_manager::LibInterface(theManager), libName(libName),
          libVersion(libVersion) {
    }

    virtual ~BehaviorSearch() {}

    // LibInterface methods

    int getLibVersion() const {return libVersion;}
    const std::string getLibName() const {return libName;}
    virtual void createModuleInfo() {}

    // BehaviorSearch methods

    /**
     * Initialize the behavior search.
     * \param numInputs number of inputs of the behavior
     * \param numOutputs number of outputs of the behavior
     * \param config YAML-based configuration the behavior search, can be empty
     */
    virtual void init(int numInputs, int numOutputs, std::string config) = 0;

    /**
     * Returns a pointer to the next behavior.
     * The BehaviorSearch retains posession of the Behavior, i.e. the
     * BehaviorSearch is responsible for cleaning up the Behavior.
     * The client must *not* call delete on the pointer.
     * The BehaviorSearch is free to dispose of the Behavior once
     * setEvaluationFeedback has been called.
     * \return behavior, must not be deleted
     */
    virtual Behavior* getNextBehavior() = 0;

    /**
     * Returns a pointer to the best evolved behavior so far.
     * The BehaviorSearch retains posession of the Behavior, i.e. the
     * BehaviorSearch is responsible for cleaning up the Behavior.
     * The client must *not* call delete on the pointer.
     * The BehaviorSearch is free to dispose of the Behavior once
     * any other interface function has been called.
     * For compatibilty reasons the function is optinal.
     * \return behavior, must not be deleted
     */
    virtual Behavior* getBestBehavior() {return NULL;}

    /**
     * Notify the BehaviorSearch of the latest Behavior's fitness
     * Calling setEvaluationFeedback invalidates the Behavior pointer
     * returned by the last call to getNextBehavior.
     * \param feedbacks feedbacks from the environment
     * \param numFeedbacks number of feedbacks
     */
    virtual void setEvaluationFeedback(const double *feedbacks,
                                       int numFeedbacks) = 0;

    /**
     * Notify the BehaviorSearch of the latest Step fitness
     * Note: This method is new and not pure abstract to keep
     * backwards compatibility.
     * \param feedbacks feedbacks from the environment
     * \param numFeedbacks number of feedbacks
     */
    virtual void setStepFeedback(const double *feedbacks,
                                 int numFeedbacks) {};

    /**
     * Notify the BehaviorSearch that a single evaluation is
     * finished and if it was aborted or successfull.
     * \param aborted = 'false' on success 'true' if evaluation
     *         was canceled
     */
    virtual void setEvaluationDone(bool aborted) {}

    /**
     * Write results to disk.
     * \param resultPath path to result files
     */
    virtual void writeResults(const std::string &resultPath) = 0;

    /**
     * Load behavior from disk.
     * \param resultPath path to result files
     * \return behavior
     */
    virtual Behavior* getBehaviorFromResults(const std::string &resultPath) = 0;

    /**
     * Check if the behavior learning is finished, e.g. it converged.
     * \return is the learning of a behavior finished?
     */
    virtual bool isBehaviorLearningDone() const = 0;

    /**
     * Check if the behavior learning can be used for parallel computing.
     * \return if the behavior search implements parallel computing.
     */
    virtual bool implementsBatchComputing() const {return false;}

    /**
     * Get a batch of behaviors to test them in parallel in the environment.
     * \return a string in yaml format with:
     *         0: string (seralized behavior)
     *         .: string (seralized behavior)
     *         .: string (seralized behavior)
     *         .: string (seralized behavior)
     *         n: string (seralized behavior)
     */
    virtual std::string getBehaviorBatch() const {return "";}
    
    /**
     * Set the feedback of a batch of behaviors.
     * \param batchFeedback array with feedback values with the size
              batchSize*numFeedbacksPerBatch
     * \param numFeedbacksPerBatch number of feedback values in the array
     *        per batch
     */
    virtual void setBatchFeedback(const double* batchFeedback, int numFeedbacksPerBatch, int batchSize) {}

    /**
     * Get behavior from serialized string.
     * \return Behavior
     */
    virtual Behavior* getBehaviorFromString(std::string &behavior) {return NULL;}

  protected:
    int numAgentInputs, numAgentOutputs;
    std::string libName;
    int libVersion;

  }; // end of class definition BehaviorSearch

} // end of namespace bolero

#endif // __BL_BEHAVIOR_SEARCH_H__
