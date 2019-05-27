cmake_minimum_required(VERSION 2.6)

project(bl_loader)
set(PROJECT_VERSION 1.0)
set(PROJECT_DESCRIPTION "Loads C++ libraries for bolero.")

enable_testing()

include(FindPkgConfig)
find_package(lib_manager)
lib_defaults()
define_module_info()

include(CheckIncludeFileCXX)
CHECK_INCLUDE_FILE_CXX("tr1/functional" USE_TR1)
if(${USE_TR1})
    ADD_DEFINITIONS(-DUSE_TR1)
else(${USE_TR1})
    ADD_DEFINITIONS(-DNO_TR1)
endif()

if(DEFINED ROCK)
    ADD_DEFINITIONS(-DROCK)
endif()

option(PYTHON_SUPPORT "Enable BLLoader to load and wrap Python modules" ON)

add_subdirectory(src)
if(PYTHON_SUPPORT)
  add_subdirectory(test)
endif(PYTHON_SUPPORT)
