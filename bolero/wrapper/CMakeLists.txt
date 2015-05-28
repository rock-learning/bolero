cmake_minimum_required (VERSION 2.6)

project(wrapper)
set(PROJECT_VERSION 1.0)
set(PROJECT_DESCRIPTION "Python wrappers for C++ interfaces.")
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

find_package(Cython REQUIRED)
set(cython_sources ${CMAKE_CURRENT_SOURCE_DIR}/_wrapper.pyx)
set(cython_output ${CMAKE_CURRENT_SOURCE_DIR}/_wrapper.cpp)

pkg_check_modules(LIB_MANAGER REQUIRED lib_manager)
pkg_check_modules(BOLERO REQUIRED bolero)
pkg_check_modules(BL_LOADER REQUIRED bl_loader)

configure_file(build_info.py.in ${CMAKE_CURRENT_SOURCE_DIR}/build_info.py)

add_custom_command(
  COMMAND ${CYTHON_EXECUTABLE}
  ARGS --cplus --fast-fail -o ${cython_output} ${cython_sources}
  DEPENDS ${cython_sources}
  OUTPUT ${cython_output})

add_custom_target(python_wrapper_cython ALL DEPENDS ${cython_output})

configure_file(test/test_library_config.txt.in ${CMAKE_CURRENT_SOURCE_DIR}/test/test_library_config.txt @ONLY)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/build_info.py DESTINATION .)
