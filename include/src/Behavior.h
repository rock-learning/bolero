/**
 * \file Behavior.h
 * \author Malte Langosz
 * \brief A first interface for a general agent.
 *
 * Version 0.1
 */

#ifndef __BL_BEHAVIOR_H__
#define __BL_BEHAVIOR_H__

#ifdef _PRINT_HEADER_
  #warning "Behaviour.h"
#endif

#include <stdexcept>
#include <cstdio>

namespace behavior_learning {

  class Behavior {

  public:
    Behavior(int numInputs, int numOutputs) : numInputs(numInputs),
                                              numOutputs(numOutputs) {}
    virtual ~Behavior() {}

    virtual Behavior* clone() {
      fprintf(stderr, "Used \"Behavior\" implementation has no \"clone()\"!\n");
      throw std::runtime_error("Used \"Behavior\" implementation has no \"clone()\"!");
    }

    virtual void setInputs(const double *values, int numInputs) = 0;
    virtual void getOutputs(double *values, int numOutputs) const = 0;

    inline int getNumInputs() const {return numInputs;}
    inline int getNumOutputs() const {return numOutputs;}

    virtual void step() = 0;

    /**
     * \return False if the Behavior has finished executing, i.e. subsequent
     *         calls to step() will result in undefined behavior.
     *         True if the Behavior can be executed for at least one more step,
     *         i.e. step() can be called at least one more time.
     *         The default implementation always returns true.
     */
    virtual bool canStep() const { return true;}

  protected:
    inline void setNumInputs(const int inputs) {numInputs = inputs;}
    inline void setNumOutputs(const int outputs) {numOutputs = outputs;}

    int numInputs, numOutputs;

  }; // end of class definition Behavior

} // end of namespace behavior_learning

#endif // __BL_BEHAVIOR_H__
