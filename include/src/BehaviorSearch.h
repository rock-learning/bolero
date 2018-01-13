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
     */
    virtual void init(int numInputs, int numOutputs) = 0;

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

  protected:
    int numAgentInputs, numAgentOutputs;
    std::string libName;
    int libVersion;

  }; // end of class definition BehaviorSearch

} // end of namespace bolero

#endif // __BL_BEHAVIOR_SEARCH_H__
