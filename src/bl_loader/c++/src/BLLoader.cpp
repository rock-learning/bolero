#ifdef PYTHON_SUPPORT
#include "PyOptimizer.h"
#include "PyBehaviorSearch.h"
#include "PyEnvironment.h"
#include "PyLoadableBehavior.h"
#include "Helper.h"
#include "PyLoadable.h"
#endif /* PYTHON_SUPPORT */
#include "LoadableBehavior.h"
#include "BLLoader.h"
#include <lib_manager/LibManager.hpp>
#include <Optimizer.h>
#include <BehaviorSearch.h>
#include <Environment.h>
#include <ContextualEnvironment.h>

#include <cstdio>
#include <stdexcept>
#include <iostream>
#include <algorithm>

using namespace std;
using lib_manager::LibManager;

namespace behavior_learning {
  namespace bl_loader {

    BLLoader::BLLoader() : lib_manager::LibInterface(new LibManager())
    {
      #ifdef PYTHON_SUPPORT
        Helper::instance().addToPyModulePath(".");
      #endif
      libManager->addLibrary(this);
    }

    BLLoader::~BLLoader() {
      libManager->releaseLibrary(getLibName());
      delete libManager;

      map<string, PyLoadable*>::iterator it;
      for(it = pythonLibraries.begin(); it != pythonLibraries.end(); ++it)
      {
        delete it->second;
      }
    }

    void BLLoader::loadConfigFile(const string &config_file) {
      libManager->loadConfigFile(config_file);
    }
    void BLLoader::loadLibrary(const string &libPath, void *config) {
      LibManager::ErrorNumber err = libManager->loadLibrary(libPath, config);
      switch(err) {
      case LibManager::LIBMGR_NO_ERROR:
        break;
      case LibManager::LIBMGR_ERR_NO_LIBRARY:
        throw runtime_error("No library named \"" + libPath + "\" exists.");
      case LibManager::LIBMGR_ERR_LIBNAME_EXISTS:
        throw runtime_error("Library \"" + libPath + "\" already exists.");
      case LibManager::LIBMGR_ERR_NOT_ABLE_TO_LOAD:
        throw runtime_error("Could not load library \"" + libPath + "\".");
      case LibManager::LIBMGR_ERR_LIB_IN_USE:
      default:
        throw runtime_error("unexpected error.");
      }
    }

    void BLLoader::addLibrary(lib_manager::LibInterface *lib) {
      libManager->addLibrary(lib);
    }

    bool BLLoader::isLibraryLoaded(const std::string &libName) {
      std::list<std::string> libNameList;
      // check for C++ libraries ...
      libManager->getAllLibraryNames(&libNameList);
      if (std::find(libNameList.begin(), libNameList.end(), libName) != libNameList.end()) {
        return true;
      }
      // check for Python libraries ...
      if (pythonLibraries.find(libName) != pythonLibraries.end()) {
        return true;
      }
      return false;
    }

    Optimizer* BLLoader::acquireOptimizer(const string &name) {
      Optimizer *lib = libManager->acquireLibraryAs<Optimizer>(name);
#ifdef PYTHON_SUPPORT
      if(!lib) {
        if(pythonLibraries.find(name) == pythonLibraries.end())
        {
          PyOptimizer* pyOptimizer = new PyOptimizer(libManager, name, 0);
          pythonLibraries[name] = pyOptimizer;
          lib = pyOptimizer;
        }
        else
        {
          lib = dynamic_cast<PyOptimizer*>(pythonLibraries[name]);
          if(!lib)
          {
            throw runtime_error("Cast failed. " + name + " is not a PyOptimizer");
          }
        }
      }
#endif
      if(!lib) {
        throw runtime_error("Could not acquire optimizer library \"" +
                            name + "\".");
      }
      return lib;
    }

    Environment* BLLoader::acquireEnvironment(const std::string &name) {
      Environment *lib = libManager->acquireLibraryAs<Environment>(name);
#ifdef PYTHON_SUPPORT
      if(!lib) {
        if(pythonLibraries.find(name) == pythonLibraries.end())
        {
          PyEnvironment* pyEnvironment =   new PyEnvironment(libManager, name, 0);
          pythonLibraries[name] = pyEnvironment;
          lib = pyEnvironment;
        }
        else
        {
          lib = dynamic_cast<PyEnvironment*>(pythonLibraries[name]);
          if(!lib)
          {
            throw runtime_error("Cast failed. " + name + " is not a PyEnvironment");
          }
        }
      }
#endif
      if(!lib) {
        throw runtime_error("Could not acquire environment library \"" +
                            name + "\".");
      }
      return lib;
    }

    ContextualEnvironment* BLLoader::acquireContextualEnvironment(const std::string &name) {
      ContextualEnvironment *lib = libManager->acquireLibraryAs<ContextualEnvironment>(name);
      if(!lib) {
        throw runtime_error("Could not acquire environment library \"" +
                            name + "\".");
      }
      return lib;
    }

    LoadableBehavior* BLLoader::acquireBehavior(const std::string &name)
    {
      LoadableBehavior *behav = libManager->acquireLibraryAs<LoadableBehavior>(name);
      #ifdef PYTHON_SUPPORT
      if(!behav)
      {
        if(pythonLibraries.find(name) == pythonLibraries.end())
        {//acquire new PyLoadableBehavior
          PyLoadableBehavior* pyBehav = new PyLoadableBehavior(libManager, name, 0);
          pythonLibraries[name] = pyBehav;
          behav = pyBehav;
        }
        else
        {//load existing behavior
          behav = dynamic_cast<PyLoadableBehavior*>(pythonLibraries[name]);
          if(!behav)
          {
            throw runtime_error("Cast failed. " + name + " is not a PyLoadableBehavior");
          }
        }
      }
      #endif
      if(!behav) {
        throw runtime_error("Could not acquire behavior library \"" +
                            name + "\".");
      }
      return behav;
    }

    BehaviorSearch* BLLoader::acquireBehaviorSearch(const std::string &name) {
      BehaviorSearch *lib = libManager->acquireLibraryAs<BehaviorSearch>(name);
#ifdef PYTHON_SUPPORT
      if(!lib) {
        if(pythonLibraries.find(name) == pythonLibraries.end())
        { //acquire new PyBehaviorSearch
          PyBehaviorSearch *pyBehaviorSearch = new PyBehaviorSearch(libManager, name, 0);
          pythonLibraries[name] = pyBehaviorSearch;
          lib = pyBehaviorSearch;
        }
        else
        {//load existing PyBehaviorSearch
          lib = dynamic_cast<PyBehaviorSearch*>(pythonLibraries[name]);
          if(!lib)
          {
            throw runtime_error("Cast failed. " + name + " is not a PyBehaviorSearch");
          }
        }
      }
#endif
      if(!lib) {
        throw runtime_error("Could not acquire behavior search library \"" +
                            name + "\".");
      }
      return lib;
    }

    void BLLoader::releaseLibrary(const string &name) {

      //Python libraries are managed by the BLLoader, C++ libs are manged by the libManager
      if(pythonLibraries.find(name) != pythonLibraries.end())
      {
        delete pythonLibraries[name];
        pythonLibraries.erase(name);
      }
      else
      {//has to be c++ behavior
        LibManager::ErrorNumber err = libManager->releaseLibrary(name);
        switch(err)
        {
          case LibManager::LIBMGR_NO_ERROR:
            break;
          case LibManager::LIBMGR_ERR_NO_LIBRARY:
            throw runtime_error("No library named \"" + name + "\" exists.");
          case LibManager::LIBMGR_ERR_LIBNAME_EXISTS:
          case LibManager::LIBMGR_ERR_NOT_ABLE_TO_LOAD:
          case LibManager::LIBMGR_ERR_LIB_IN_USE:
          default:
            throw runtime_error("unexpected error.");
        }

        //FIXME according to the documentation of the LibManager this call should not be necessary (see https://github.com/rock-simulation/mars/issues/5)
        err = libManager->unloadLibrary(name);
        switch(err)
        {
          case LibManager::LIBMGR_NO_ERROR:
          case LibManager::LIBMGR_ERR_LIB_IN_USE:
            break;
          case LibManager::LIBMGR_ERR_NO_LIBRARY:
            throw runtime_error("No library named \"" + name + "\" exists.");
          default:
            throw runtime_error("unexpected error.");
        }
      }
    }

    void BLLoader::dumpTo(const std::string &file) {
      libManager->dumpTo(file);
    }

  void BLLoader::unloadLibrary(const std::string &name) {
    std::cerr << "BLLoader::unloadLibrary() is deprecated and should not be used anymore. It is sufficient to call BLLoader::releaseLibrary(). I.e. you can simply remove the call to unloadLibrary().";
  }
  } // end of namespace bl_loader
} // end of namespace behavior_learning
