cmake_minimum_required(VERSION 2.6)

include(FindPkgConfig)

# add the binary tree to the search path for include files
include_directories("${PROJECT_BINARY_DIR}")

set(BL_LOADER_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src CACHE STRING "Include dir for bl_loader.")
set(BL_LOADER_LIBRARY_DIRS ${CMAKE_CURRENT_BINARY_DIR} CACHE STRING "Lib directory for bl_loader.")

pkg_check_modules(LIB_MANAGER "lib_manager")
include_directories(${LIB_MANAGER_INCLUDE_DIRS})
link_directories(${LIB_MANAGER_LIBRARY_DIRS})

pkg_check_modules(BOLERO "bolero")
include_directories(${BOLERO_INCLUDE_DIRS})
link_directories(${BOLERO_LIBRARY_DIRS})

pkg_check_modules(YAML "yaml-cpp")
include_directories(${YAML_INCLUDE_DIRS})
link_directories(${YAML_LIBRARY_DIRS})

pkg_check_modules(CONFIGMAPS "configmaps")
include_directories(${CONFIGMAPS_INCLUDE_DIRS})
link_directories(${CONFIGMAPS_LIBRARY_DIRS})

include_directories(src)

set(SOURCES src/BLLoader.cpp)
set(HEADERS src/BLLoader.h)

if(PYTHON_SUPPORT)
  if("$ENV{PYTHON}" STREQUAL "")
      set(PYTHON "python")
      message(STATUS "Using default python.")
  else()
      set(PYTHON $ENV{PYTHON})
  endif()
  execute_process(
      COMMAND which ${PYTHON}
      OUTPUT_VARIABLE PYTHON_EXECUTABLE
  )
  execute_process(
      COMMAND ${PYTHON} -c "import sys;print(sys.version_info.major)"
      OUTPUT_VARIABLE PYTHON_MAJOR_VERSION
  )
  execute_process(
      COMMAND ${PYTHON} -c "import sys;print(sys.version_info.minor)"
      OUTPUT_VARIABLE PYTHON_MINOR_VERSION
  )
  string(STRIP ${PYTHON_MAJOR_VERSION} PYTHON_MAJOR_VERSION)
  string(STRIP ${PYTHON_MINOR_VERSION} PYTHON_MINOR_VERSION)
  message(STATUS "Python: ${PYTHON}")
  message(STATUS "Python executable: ${PYTHON_EXECUTABLE}")
  message(STATUS "Python major version: ${PYTHON_MAJOR_VERSION}")

  add_definitions(-DPYTHON_VERSION=${PYTHON_MAJOR_VERSION})

  add_definitions(-DPYTHON_SUPPORT)
  add_definitions(-DUSE_MEMORYVIEWS)

  if (APPLE)
    find_package(PythonLibs ${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION} EXACT REQUIRED)
  elseif(${PYTHON_MAJOR_VERSION} EQUAL 3)
    find_package(PythonLibs ${PYTHON_MAJOR_VERSION} EXACT REQUIRED)
  else()
    # HACK find_package is not able to find Python 2 include directories
    pkg_check_modules(PYTHON ${PYTHON})
  endif()
  message(STATUS "Python include directories: ${PYTHON_INCLUDE_DIRS}")
  include_directories(${PYTHON_INCLUDE_DIRS})

  execute_process(
    COMMAND ${PYTHON} -c "import numpy; print(numpy.get_include())"
    RESULT_VARIABLE NUMPY_RESULT
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS
    ERROR_VARIABLE NUMPY_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  if(${NUMPY_RESULT} EQUAL 0)
    message(STATUS "Numpy include directories: ${NUMPY_INCLUDE_DIRS}")
  else()
    message(FATAL_ERROR "NumPy import failure:\n${NUMPY_ERROR}")
  endif()
  include_directories(${NUMPY_INCLUDE_DIRS})
  link_directories(${PYTHON_LIBRARY_DIRS})

  set(SOURCES
    ${SOURCES}
    src/PyOptimizer.cpp
    src/PyEnvironment.cpp
    src/PyBehavior.cpp
    src/PyBehaviorSearch.cpp
    src/PyLoadableBehavior.cpp
    src/PythonInterpreter.cpp
  )
  set(HEADERS
    ${HEADERS}
    src/PyOptimizer.h
    src/PyEnvironment.h
    src/PyBehavior.h
    src/PyBehaviorSearch.h
    src/PyLoadableBehavior.h
    src/PyLoadable.h
    src/PythonInterpreter.hpp
  )
if(WIN32)
set(EXTRA_LIBRARIES
    ${EXTRA_LIBRARIES}
    ${PYTHON_LIBRARIES}
  )
else(WIN32)
set(EXTRA_LIBRARIES
    ${EXTRA_LIBRARIES}
    ${PYTHON_LIBRARIES}
    -lutil
  )
endif()
endif()

add_library(${PROJECT_NAME} SHARED ${SOURCES})
add_library(${PROJECT_NAME}_static STATIC ${SOURCES})
target_link_libraries(${PROJECT_NAME}
                      ${LIB_MANAGER_LIBRARIES}
                      ${EXTRA_LIBRARIES}
                      ${CONFIGMAPS_LIBRARIES}
                      ${YAML_LIBRARIES})

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

# Install the library into the lib folder
install(TARGETS ${PROJECT_NAME} ${_INSTALL_DESTINATIONS})
install(TARGETS ${PROJECT_NAME}_static ${_INSTALL_DESTINATIONS})

CONFIGURE_FILE(${PROJECT_NAME}.pc.in ${PROJECT_NAME}.pc @ONLY)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc DESTINATION lib/pkgconfig)

# Install headers into mars include directory
install(FILES ${HEADERS} DESTINATION include/bolero/${PROJECT_NAME})
