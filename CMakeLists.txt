cmake_minimum_required(VERSION 2.6)

enable_testing()

if("$ENV{PYTHON}" STREQUAL "")
    set(PYTHON "python")
    message(STATUS "Using default python.")
else()
    set(PYTHON $ENV{PYTHON})
    message(STATUS ${PYTHON})
endif()

if("$ENV{CYTHON}" STREQUAL "")
    set(CYTHON "cython")
    message(STATUS "Using default cython.")
else()
    set(CYTHON $ENV{CYTHON})
    message(STATUS ${CYTHON})
endif()

execute_process(
COMMAND ${PYTHON} -c "import sys;print(sys.version_info.major)"
OUTPUT_VARIABLE PYTHON_VERSION
)
string(STRIP ${PYTHON_VERSION} PYTHON_VERSION)
message(STATUS "Python version: ${PYTHON_VERSION}")


execute_process(
COMMAND ${CYTHON} -${PYTHON_VERSION} bolero/utils/_ranking_svm.pyx
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set(SETUP_PY "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")

# Custom python target that runs 'setup.py build' to build all python
# components.
add_custom_target(python_setup ALL)
add_custom_command(
    TARGET python_setup
    PRE_BUILD
    COMMAND ${PYTHON}
    ARGS ${SETUP_PY} build_ext -i
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

install(
  CODE "execute_process(
    COMMAND ${PYTHON} ${SETUP_PY} install --prefix=${CMAKE_INSTALL_PREFIX}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})"
)

# nose is the python unit testing program it will run all python unit tests
add_test(nosetests nosetests ${CMAKE_CURRENT_SOURCE_DIR} -sv)
