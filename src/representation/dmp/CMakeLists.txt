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
pkg_check_modules(EIGEN3 REQUIRED eigen3)
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp)

add_subdirectory(src)
include_directories(src)

configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/dmp_cpp.pc.in"
  "${PROJECT_BINARY_DIR}/dmp_cpp.pc" @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/dmp_cpp.pc DESTINATION lib/pkgconfig)

configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/quaternion_dmp_cpp.pc.in"
    "${PROJECT_BINARY_DIR}/quaternion_dmp_cpp.pc" @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/quaternion_dmp_cpp.pc DESTINATION lib/pkgconfig)

configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/rigid_body_dmp_cpp.pc.in"
    "${PROJECT_BINARY_DIR}/rigid_body_dmp_cpp.pc" @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/rigid_body_dmp_cpp.pc DESTINATION lib/pkgconfig)

find_package(Cython REQUIRED)
configure_file(
    python/dmp/build_info.py.in
    ${CMAKE_CURRENT_SOURCE_DIR}/python/dmp/build_info.py)

enable_testing()
add_subdirectory(test_dmp_behavior)
add_subdirectory(test_dmp_quaternion_behavior)
add_subdirectory(test)

# Custom python target that runs 'setup.py build' to build all python
# components.
option(PYTHON_SUPPORT "Build and install python wrapper" ON)
find_program(PYTHON "python")
if (PYTHON_SUPPORT AND PYTHON)
  set(SETUP_PY "${CMAKE_CURRENT_SOURCE_DIR}/python/dmp/setup.py")

  add_custom_target(python_setup ALL DEPENDS dmp_cpp rigid_body_dmp_cpp)
  add_custom_command(
    TARGET python_setup
    PRE_BUILD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
    COMMAND ${PYTHON}
    ARGS ${SETUP_PY} build_ext -i)

    install(
      CODE "execute_process(
        COMMAND ${PYTHON} ${SETUP_PY} install --prefix=${CMAKE_INSTALL_PREFIX}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python)")

    # nose is the python unit testing program it will run all python unit tests
    add_test(nosetests nosetests ${CMAKE_CURRENT_SOURCE_DIR} -sv)
elseif(PYTHON_SUPPORT)
    message(FATAL_ERROR "Python executable not found")
endif()
