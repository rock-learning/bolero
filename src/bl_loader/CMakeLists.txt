cmake_minimum_required(VERSION 2.6)
project(bl_loader)
set(PROJECT_VERSION 1.0)
set(PROJECT_DESCRIPTION "A helper plugin to load c++-libraries for bolero.")

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

option(PYTHON_SUPPORT "Enable BLLoader to load and wrap Python modules" ON)

add_subdirectory(c++)
if(PYTHON_SUPPORT)
  #the test needs to know whether it is build at the same time as the bl_loader or not.
  #If it is build at the same time, it will link against and include the current bl_loader,
  #Otherwise the system bl_loader will be used.
  set(BUILD_TEST_AS_PART_OF_BL_LOADER TRUE)
  add_subdirectory(test)
endif(PYTHON_SUPPORT)
