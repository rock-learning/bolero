cmake_minimum_required (VERSION 2.6)

project(dmp)
set(PROJECT_VERSION 1.0)
set(PROJECT_DESCRIPTION "Dynamical Movement Primitives.")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(FindPkgConfig)
find_package(lib_manager)
lib_defaults()
define_module_info()

IF (WIN32)
  SET (POSTLIB ".dll")
  SET (PRELIB "lib")
  SET (LIBPATH "bin")
  SET (LIBRARY_PATH_VAR "LD_LIBRARY_PATH")
ELSE (WIN32)
  IF (APPLE)
    SET (POSTLIB ".dylib")
    SET (PRELIB "lib")
    SET (LIBPATH "lib")
    SET (LIBRARY_PATH_VAR "DYLD_LIBRARY_PATH")
  ELSE (APPLE)
    SET (POSTLIB ".so")
    SET (PRELIB "lib")
    SET (LIBPATH "lib")
    SET (LIBRARY_PATH_VAR "LD_LIBRARY_PATH")
  ENDIF (APPLE)
ENDIF (WIN32)

add_definitions ("-Wall")

# Define variables used in build_info.py.in
pkg_check_modules(LIB_MANAGER REQUIRED lib_manager)
pkg_check_modules(BOLERO REQUIRED bolero)
pkg_check_modules(BL_LOADER REQUIRED bl_loader)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(Eigen3 3.2.4 REQUIRED)
find_package(YamlCpp REQUIRED)

add_subdirectory(src)
include_directories(src)

find_package(Cython REQUIRED)
configure_file(build_info.py.in ${CMAKE_CURRENT_SOURCE_DIR}/build_info.py)

# Custom python target that runs 'setup.py build' to build all python
# components.
option(PYTHON_SUPPORT "Build and install python wrapper" ON)
find_program(PYTHON "python")
if (PYTHON_SUPPORT AND PYTHON)
  set(SETUP_PY "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")

  add_custom_target(python_setup ALL)
  add_custom_command(
    TARGET python_setup
    PRE_BUILD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${PYTHON}
    ARGS ${SETUP_PY} build_ext -i)

    install(
      CODE "execute_process(
        COMMAND ${PYTHON} ${SETUP_PY} install --prefix=${CMAKE_INSTALL_PREFIX}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})")

    # nose is the python unit testing program it will run all python unit tests
    add_test(nosetests nosetests ${CMAKE_CURRENT_SOURCE_DIR} -sv)
elseif(PYTHON_SUPPORT)
    message(FATAL_ERROR "Python executable not found")
endif()

enable_testing()
#add_subdirectory(test_dmp_behavior)
#add_subdirectory(test_dmp_quaternion_behavior)
#add_subdirectory (test)
