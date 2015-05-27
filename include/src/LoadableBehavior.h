/**
 * @file LoadableBehavior.h
 *  @author Arne Boeckmann (arne.boeckmann@dfki.de)
 */


#ifndef LOADABLEBEHAVIOR_H_
#define LOADABLEBEHAVIOR_H_

#ifdef _PRINT_HEADER_
  #warning "LoadableBehavior.h"
#endif

#include <lib_manager/LibInterface.hpp>
#include "Behavior.h"
#include <string>

namespace bolero {

  /**
   * A Behavior that can be loaded by the the bl_loader.
   */
  class LoadableBehavior : public lib_manager::LibInterface, public Behavior {

  public:
    LoadableBehavior(lib_manager::LibManager *libManager,
                const std::string &libName, const int libVersion,
                const int numInputs, const int numOutputs) :
      lib_manager::LibInterface(libManager),
      Behavior(numInputs, numOutputs),
      libName(libName),
      libVersion(libVersion)
    {}

    virtual ~LoadableBehavior() {}

    // LibInterface methods
    virtual int getLibVersion() const {return libVersion;}
    virtual const std::string getLibName() const {return libName;}
    virtual void createModuleInfo() {}

    /**Initializes the Behavior using the specified config file.
     * Initialization should be done only once. It is intended to set the
     * Behavior's template parameters.
     *
     * The Behavior needs to be initialized before any other method can be called.
     * 
     * \return true if the initialization was successful, false otherwise.
     *
     */
    virtual bool initialize(const std::string& initialConfigPath) = 0;

    /**Configures the Behavior using the specified config file.
     * This method should be called after initialize().
     * It sets the runtime parameters of the Behavior and may be called
     * multiple times during execution.
     *
     * The config file should be in yaml format.
     * \return true if the configuration was successful, false otherwise.
     */
    virtual bool configure(const std::string& configPath) = 0;

    /** Configures the behavior using the specified \p yaml string.
     *  The string should contain valid yaml.
     *
     * \return true if the configuration was successful, false otherwise.
     */
    virtual bool configureYaml(const std::string& yaml) = 0;


  protected:
    std::string libName;
    int libVersion;

  }; // end of class definition LoadableBehavior

} // end of namespace bolero

#endif /* LOADABLEBEHAVIOR_H_ */
