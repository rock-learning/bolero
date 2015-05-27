#ifndef BL_LOADER_H
#define BL_LOADER_H

#include <lib_manager/LibManager.hpp>
#include <string>
#include <vector>
#include <map>



namespace lib_manager {
  class LibManager;
}

namespace bolero {
  class Optimizer;
  class BehaviorSearch;
  class Environment;
  class ContextualEnvironment;
  class LoadableBehavior;

  namespace bl_loader {
    class PyLoadable;
  /**
    * Loads components at runtime.
    *
    * \note The library names are unique. I.e. if you acquire a behavior named "bla" you cannot acquire an optimizer
    *       with the same name. Trying this will result in a std::runtime_error.
    */
    class BLLoader : public lib_manager::LibInterface{
    public:
      BLLoader();
      ~BLLoader();

      // LibInterface methods
      virtual int getLibVersion() const {return 0;}
      virtual const std::string getLibName() const {return "bl_loader";}

      CREATE_MODULE_INFO();

      void loadConfigFile(const std::string &config_file);

      //FIXME what does this method do? Is it used?
      void loadLibrary(const std::string &libPath, void *config=NULL);
      void addLibrary(lib_manager::LibInterface *lib);
      bool isLibraryLoaded(const std::string &libName);

      /**
      * \note The BlLoader is responsible for the memory management of all returned pointers.
      *       Do NOT delete them manually.
      *
      * \throws std::runtime_error in case of error
      */
      Optimizer* acquireOptimizer(const std::string &name);

      /**
      * \note The BlLoader is responsible for the memory management of all returned pointers.
      *       Do NOT delete them manually.
      *
      * \throws std::runtime_error in case of error
      */
      BehaviorSearch* acquireBehaviorSearch(const std::string &name);

      /**
      * \note The BlLoader is responsible for the memory management of all returned pointers.
      *       Do NOT delete them manually.
      *
      * \throws std::runtime_error in case of error
      */
      Environment* acquireEnvironment(const std::string &name);
      ContextualEnvironment* acquireContextualEnvironment(const std::string &name);

      /**
       * Loads the behavior specified by \p name.
       * If a c++ implementation of the behavior is found it will be acquired.
       * Otherwise the python implementation of the behavior will be acquired.
       *
       * \note Generally the configuration files for python behaviors
       *       differ from their c++ counter parts. Thus the user needs
       *       to know whether the requested behavior is implemented in
       *       python or c++. There is currently no workaround for this bug.
       *
       * \note If PYTHON_SUPPORT is enabled, this method cannot fail.
       *       If no c++ implementation of the requested behavior is found
       *       a generic lazy loading python behavior will be returned.
       *       This python behavior will fail upon initialization if there
       *       is no python behavior with the specified name.
       *       This sucks but there is no way to check if the python behavior
       *       actually exists from inside this method.
       *
       * \param name The name of the behavior that should be loaded.
       *
       *
       * \return If a behavior with the same name has already been loaded, it will be returned.
       *         Otherwise a new behavior will be loaded and returned.
       *         The returned pointer is managed by the BLLoader. The memory ca be freed by calling releaseLibrary()
       *         or by destroying the BLLoader. Do NOT free it yourself.
       */
      LoadableBehavior* acquireBehavior(const std::string &name);

      /**
      * Releases the Optimizer, BehaviorSearch, Environment, or Behavior that has been acquired
      * using \p name.
      * After calling this method, all pointers pointing to the library will be invalid. I.e. using them will crash
      * your program.
      *
      * \throws std::runtime_error if the library does not exist
      */
      void releaseLibrary(const std::string &name);

      /**
      * This method is deprecated and should not be used.
      * It is sufficient to call BLLoader::releaseLibrary().
      * I.e. you can simply remove calls to unloadLibrary().
      */
      void unloadLibrary(const std::string &name);

      void dumpTo(const std::string &file);

    private:

      BLLoader(const BLLoader&);
      BLLoader& operator=(const BLLoader&);

      /**Contains all python libs that have been acquired.*/
      std::map<std::string, PyLoadable*> pythonLibraries;

    }; // end of class BLLoader

  } // end of namespace bl_loader
} // end of namespace bolero

#endif /* BL_LOADER_H */

