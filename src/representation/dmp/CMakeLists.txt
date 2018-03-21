cmake_minimum_required (VERSION 2.6)

project(dmp)
set(PROJECT_VERSION 1.0)
set(PROJECT_DESCRIPTION "Dynamical Movement Primitives.")

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

add_subdirectory(implementation)
add_subdirectory(src)
include_directories(src ${LIB_MANAGER_INCLUDE_DIRS})

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

enable_testing()
add_subdirectory(test_dmp_behavior)
add_subdirectory(test_dmp_quaternion_behavior)
add_subdirectory(test_dmp_rigid_body_behavior)
add_subdirectory(test)
