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
#include <map>
#include <string>
#include <vector>


namespace bolero {

  /**
   * @class Behavior
   * Behavior interface.
   *
   * A behavior maps input (e.g. state) to output (e.g. next state or action).
   */
  class Behavior {

  public:
    Behavior() : numInputs(-1), numOutputs(-1) {}
    virtual ~Behavior() {}

    /**
     * Initialize the behavior.
     * \param numInputs number of inputs
     * \param numOutputs number of outputs
     */
    virtual void init(int numInputs, int numOutputs)
    {
        this->numInputs = numInputs;
        this->numOutputs = numOutputs;
    }

    /**
     * Clone behavior.
     * \throw std::runtime_error if not overwritten by subclass
     * \return cloned behavior, has to be deleted
     */
    virtual Behavior* clone() {
      fprintf(stderr, "Used \"Behavior\" implementation has no \"clone()\"!\n");
      throw std::runtime_error("Used \"Behavior\" implementation has no \"clone()\"!");
    }

    /**
     * Set input for the next step.
     * If the input vector consists of positions and derivatives of these,
     * by convention all positions and all derivatives should be stored
     * contiguously.
     * \param values inputs e.g. current state of the system
     * \param numInputs number of inputs
     */
    virtual void setInputs(const double *values, int numInputs) = 0;

    /**
     * Set inputs observerd after performing a step (next state).
     * If the input vector consists of positions and derivatives of these,
     * by convention all positions and all derivatives should be stored
     * contiguously.
     * \param values inputs e.g. current state of the system
     * \param numInputs number of inputs
     */
    virtual void setTargetState(const double *values, int numInputs) {};

    /**
     * Get outputs of the last step.
     * If the output vector consists of positions and derivatives of these,
     * by convention all positions and all derivatives should be stored
     * contiguously.
     * \param[out] values outputs, e.g. desired state of the system
     * \param numOutputs expected number of outputs
     */
    virtual void getOutputs(double *values, int numOutputs) const = 0;

    /**
     * Get number of inputs.
     * \return number of inputs
     */
    inline int getNumInputs() const {return numInputs;}

    /**
     * Get number of outputs.
     * \return number of outputs
     */
    inline int getNumOutputs() const {return numOutputs;}

    /**
     * Compute output for the received input.
     * Uses the inputs and meta-parameters to compute the outputs.
     */
    virtual void step() = 0;
    /**
     * Returns if step() can be called again.
     * \return False if the Behavior has finished executing, i.e. subsequent
     *         calls to step() will result in undefined behavior.
     *         True if the Behavior can be executed for at least one more step,
     *         i.e. step() can be called at least one more time.
     *         The default implementation always returns true.
     */

    /**
     * Is called after the step is applied in the environmet.
     */
    virtual void finishStep() {}

    virtual bool canStep() const { return true;}

    /**
     * Meta-parameters could be the goal, obstacles, etc.
     * Each parameter is a list of doubles identified by a key.
     */
    typedef std::map<std::string, std::vector<double> > MetaParameters;

    /**
     * Set meta-parameters.
     * Meta-parameters could be the goal, obstacles, ...
     * \throw std::runtime_error if not overwritten by subclass
     * \param params meta-parameters
     */
    virtual void setMetaParameters(const MetaParameters &params) {
        throw std::runtime_error("Used \"Behavior\" implementation has no "
                                 "\"setMetaParameters()\"!");
    }

  protected:
    inline void setNumInputs(const int inputs) {numInputs = inputs;}
    inline void setNumOutputs(const int outputs) {numOutputs = outputs;}

    int numInputs, numOutputs;

  }; // end of class definition Behavior

} // end of namespace bolero

#endif // __BL_BEHAVIOR_H__
