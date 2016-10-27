configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")

# Custom python target that runs 'setup.py build' to build all python
# components.
option(PYTHON_SUPPORT "Build and install python wrapper" ON)
find_program(PYTHON "python")
if (PYTHON_SUPPORT AND PYTHON)
  find_package(Cython REQUIRED)

  set(SETUP_PY "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")

  add_custom_target(python_setup ALL DEPENDS dmp)
  add_custom_command(
    TARGET python_setup
    PRE_BUILD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${PYTHON}
    ARGS ${SETUP_PY} build_ext -i
  )

    install(
      CODE "execute_process(
        COMMAND ${PYTHON} ${SETUP_PY} install --prefix=${CMAKE_INSTALL_PREFIX}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})"
    )

    # nose is the python unit testing program it will run all python unit tests
    add_test(
      NAME nosetests
      COMMAND nosetests
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
elseif(PYTHON_SUPPORT)
    message(FATAL_ERROR "Python executable not found")
endif()
