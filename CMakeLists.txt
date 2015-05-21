cmake_minimum_required (VERSION 2.6)

project(bolero)
set(PROJECT_VERSION 0.0)
set(PROJECT_DESCRIPTION "Behavior Optimization and Learning for Robots")

set(CMAKE_COLOR_MAKEFILE ON)

# Used by some sub projects to determine whether they are build as a sub
# project of bolero or not
set(BUILD_BOLERO TRUE)

# The packageList.txt file contains all subdirectories that should be build
# if it is not present we cannot continue
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/packageList.txt)
	message(FATAL_ERROR
	    "'packageList.txt' is missing. Probably you need to remove the cmake "
	    "build files before calling cmake again. Otherwise it might ignore "
	    "the added define.")
endif()

# This configure rule is used to trigger a reconfigure whenever the packageList
# changes
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/packageList.txt packageList.txt.tmp COPYONLY)

# Read required sub projects from packageList.txt
file(READ packageList.txt SUB_PROJECTS)
# Remove comments from the packageList.txt (everything between # and line end)
string(REGEX REPLACE "\#[^\n]*\n" "" SUB_PROJECTS "${SUB_PROJECTS}")
# Convert file content into a valid cmake list (stolen from
# http://www.cmake.org/pipermail/cmake/2007-May/014222.html)
STRING(REGEX REPLACE ";" "\\\\;" SUB_PROJECTS "${SUB_PROJECTS}")
STRING(REGEX REPLACE "\n" ";" SUB_PROJECTS "${SUB_PROJECTS}")

# Needs to be enabled here because sub projects may use add_test()
enable_testing()

# Macros for passing lists between different directories through an internal
# cache variable (from
# http://www.cmake.org/pipermail/cmake/2007-November/018109.html). They are
# used to dynamically build a list of targets.
MACRO (APPEND_INTERNAL_LIST  LIST_NAME  VALUE)
   # If the list in not in the cache, create it.
   IF (${LIST_NAME})
      SET (${LIST_NAME} "${${LIST_NAME}};${VALUE}" CACHE INTERNAL "Internal
variable")
   ELSE (${LIST_NAME})
      SET (${LIST_NAME} "${VALUE}" CACHE INTERNAL "Internal variable")
   ENDIF (${LIST_NAME})
ENDMACRO (APPEND_INTERNAL_LIST)

MACRO (INITIALIZE_INTERNAL_LIST  LIST_NAME)
   SET (${LIST_NAME} "" CACHE INTERNAL "Internal variable")
ENDMACRO (INITIALIZE_INTERNAL_LIST)

# Override add_library and add_executable to create a list of all targets at
# runtime. This is a crude hack but there is no other way to dynamically
# generate a list of all targets. The list is used as dependency list for the
# 'python_setup' target. This is done because some targets generate python
# binding code at compile time. Those generated files need to be there when
# setup.py is run...
INITIALIZE_INTERNAL_LIST(allTargets) # Initialize to clear any values from last run
macro(add_library _target)
    _add_library (${_target} ${ARGN})
    APPEND_INTERNAL_LIST(allTargets ${_target})
endmacro()

macro(add_executable _target)
    _add_executable (${_target} ${ARGN})
    APPEND_INTERNAL_LIST(allTargets ${_target})
endmacro()

macro(add_custom_target _target)
    _add_custom_target (${_target} ${ARGN})
    APPEND_INTERNAL_LIST(allTargets ${_target})
endmacro()

# Add sub projects
FOREACH(subdir ${SUB_PROJECTS}) 
    add_subdirectory(${subdir})
ENDFOREACH()

# There is a nasty bug when using FindQt4 (which is used by most of the
# environments). FindQt4 adds targets named 'Qt4::blabla'. However those
# targets cannot be used as dependency because targets containing colons
# break make. This loop copies all targets containing no colons to a new list
set(filteredTargetList "")
FOREACH(target ${allTargets}) 
    if(NOT ${target} MATCHES ":")
        LIST(APPEND filteredTargetList ${target})
    endif()
ENDFOREACH()

# Custom python target that runs 'setup.py build' to build all python
# components. This target depends on all sub projects (because some of them
# may generate python wrappers)
option(WithPython "Build and install python components as well" ON)
find_program(PYTHON "python")
if (WithPython AND PYTHON)
    set(SETUP_PY "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")

    add_custom_target(python_setup ALL
                      # Python should be built after all sub projects because
                      # some sub projectes generate python extension code
                      DEPENDS ${filteredTargetList})

    # This command will be executed before target 'python_setup' is build.
    add_custom_command(TARGET python_setup
                       PRE_BUILD
                       # Needs to be set because setup.py contains relativ paths
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                       COMMAND ${PYTHON}
                       ARGS ${SETUP_PY} build_ext -i)

    install(CODE
        "execute_process(
            COMMAND ${PYTHON} ${SETUP_PY} install --prefix=${CMAKE_INSTALL_PREFIX}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})")
elseif(WithPython)
    message(FATAL_ERROR "Python executable not found")
endif()

# nose is the python unit testing program it will run all python unit tests
add_test(nosetests nosetests ${CMAKE_CURRENT_SOURCE_DIR} -sv)
