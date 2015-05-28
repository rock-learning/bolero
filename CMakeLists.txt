cmake_minimum_required(VERSION 2.6)

enable_testing()
add_subdirectory("include")

# Custom python target that runs 'setup.py build' to build all python
# components.
option(PYTHON_SUPPORT "Build and install python components as well" ON)
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
