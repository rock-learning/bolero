cmake_minimum_required(VERSION 2.6)

project(test_py_loadable_behavior)
set(PROJECT_VERSION 1.0)
set(PROJECT_DESCRIPTION "")

include(FindPkgConfig)
find_package(lib_manager)
lib_defaults()
define_module_info()

pkg_check_modules(LIB_MANAGER REQUIRED lib_manager)
include_directories(${LIB_MANAGER_INCLUDE_DIRS})
link_directories(${LIB_MANAGER_LIBRARY_DIRS})

pkg_check_modules(CATCH REQUIRED catch)
include_directories(${CATCH_INCLUDE_DIRS})

pkg_check_modules(BOLERO REQUIRED bolero)
include_directories(${BOLERO_INCLUDE_DIRS})

pkg_check_modules(CONFIGMAPS configmaps)
include_directories(${CONFIGMAPS_INCLUDE_DIRS})
link_directories(${CONFIGMAPS_LIBRARY_DIRS})

include_directories(${BL_LOADER_INCLUDE_DIRS})
link_directories(${BL_LOADER_LIBRARY_DIRS})

include_directories(src)

if(UNIX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} bl_loader ${CONFIGMAPS_LIBRARIES})

file(COPY python/functions.py DESTINATION .)

if(WIN32)
  set(LIB_INSTALL_DIR bin) # .dll are in PATH, like executables
else(WIN32)
  set(LIB_INSTALL_DIR lib)
endif(WIN32)

set(_INSTALL_DESTINATIONS
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION ${LIB_INSTALL_DIR}
	ARCHIVE DESTINATION lib
)

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

install(TARGETS ${PROJECT_NAME} ${_INSTALL_DESTINATIONS})

configure_file(config/init.yaml.in ${CMAKE_CURRENT_BINARY_DIR}/init.yaml @ONLY)
configure_file(config/config.yaml.in ${CMAKE_CURRENT_BINARY_DIR}/config.yaml @ONLY)
configure_file(config/learning_config.yml.in ${CMAKE_CURRENT_BINARY_DIR}/learning_config.yml @ONLY)
configure_file(python/test_behavior.py ${CMAKE_CURRENT_BINARY_DIR}/test_behavior.py @ONLY)

add_test(test_py_loadable_behavior test_py_loadable_behavior)
