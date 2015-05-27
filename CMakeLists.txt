cmake_minimum_required (VERSION 2.6)

project(bolero)
set(PROJECT_VERSION 0.1)
set(PROJECT_DESCRIPTION "Behavior Optimization and Learning for Robots")
set(CMAKE_COLOR_MAKEFILE ON)

# Needs to be enabled here because sub projects may use add_test()
enable_testing()

add_subdirectory("include")

# Custom python target that runs 'setup.py build' to build all python
# components.
option(WithPython "Build and install python components as well" ON)
find_program(PYTHON "python")
if (WithPython AND PYTHON)
    set(SETUP_PY "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")

    add_custom_target(python_setup ALL)

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
