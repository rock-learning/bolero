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

namespace behavior_learning {

  class BehaviorSearch : public lib_manager::LibInterface {

  public:
    BehaviorSearch(lib_manager::LibManager *theManager,
                   const std::string &libName,
                   int libVersion) :
      lib_manager::LibInterface(theManager), libName(libName),
      libVersion(libVersion) {
    }

    virtual ~BehaviorSearch() {}

    // LibInterface methods
    int getLibVersion() const {return libVersion;}
    const std::string getLibName() const {return libName;}
    virtual void createModuleInfo() {}

    // BehaviorSearch methods
    virtual void init(int numInputs, int numOutputs) = 0;

    /**
     * @brief returns a pointer to the next behavior.
     * The BehaviorSearch retains posession of the Behavior, i.e. the
     * BehaviorSearch is responsible for cleaning up the Behavior.
     * The client must *not* call delete on the pointer.
     * The BehaviorSearch is free to dispose of the Behavior once
     * setEvaluationFeedback has been called.
     */
    virtual Behavior* getNextBehavior() = 0;

    /**
     * @brief returns a pointer to the best evolved behavior so far.
     * The BehaviorSearch retains posession of the Behavior, i.e. the
     * BehaviorSearch is responsible for cleaning up the Behavior.
     * The client must *not* call delete on the pointer.
     * The BehaviorSearch is free to dispose of the Behavior once
     * any other interface function has been called.
     * For compatibilty reasons the function is optinal.
     */
    virtual Behavior* getBestBehavior() {return NULL;}

    /**
     * @brief notify the BehaviorSearch of the latest Behavior's fitness
     * Calling setEvaluationFeedback invalidates the Behaivor pointer
     * returned by the last call to getNextBehavior.
     */
    virtual void setEvaluationFeedback(const double *feedbacks,
                                       int numFeedbacks) = 0;
    virtual void writeResults(const std::string &resultPath) = 0;
    virtual Behavior* getBehaviorFromResults(const std::string &resultPath) = 0;
    virtual bool isBehaviorLearningDone() const = 0;

  protected:
    int numAgentInputs, numAgentOutputs;
    std::string libName;
    int libVersion;

  }; // end of class definition BehaviorSearch

} // end of namespace behavior_learning

#endif // __BL_BEHAVIOR_SEARCH_H__
