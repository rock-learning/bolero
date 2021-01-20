if(DEFINED PYTHON_EXECUTABLE)
    message(STATUS "Using python from CMake variable.")
elseif("$ENV{PYTHON}" STREQUAL "")
    set(PYTHON_EXECUTABLE "python")
    message(STATUS "Using default python.")
else()
    set(PYTHON_EXECUTABLE $ENV{PYTHON})
    message(STATUS "Using python from environment variable.")
endif()
message(STATUS "Python executable: ${PYTHON_EXECUTABLE}")

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sys;print(sys.version_info.major)"
    OUTPUT_VARIABLE PYTHON_VERSION
)
string(STRIP ${PYTHON_VERSION} PYTHON_VERSION)
message(STATUS "Python version: ${PYTHON_VERSION}")


configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")

# Custom python target that runs 'setup.py build' to build all python
# components.
option(PYTHON_SUPPORT "Build and install python wrapper" ON)
if (PYTHON_SUPPORT)

    set(SETUP_PY "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
    add_custom_target(python_setup ALL)
    add_custom_command(
        TARGET python_setup
        PRE_BUILD
        COMMAND ${PYTHON_EXECUTABLE}
        ARGS ${SETUP_PY} build_ext -i
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    install(
      CODE "execute_process(
        COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install --prefix=${CMAKE_INSTALL_PREFIX}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})"
    )

    # nose is the python unit testing program it will run all python unit tests
    add_test(
      NAME nosetests
      COMMAND ${PYTHON_EXECUTABLE} -m nose
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
elseif(PYTHON_SUPPORT)
    message(FATAL_ERROR "Python executable not found")
endif()
