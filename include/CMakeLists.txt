project(bolero)
set(PROJECT_VERSION 1.0)
set(PROJECT_DESCRIPTION "This is a header only library which includes interfaces for bolero.")
include(FindPkgConfig)

cmake_minimum_required(VERSION 2.6)

set(HEADERS
        src/Behavior.h
        src/BehaviorSearch.h
        src/Optimizer.h
        src/ContextualEnvironment.h
        src/Environment.h
        src/LoadableBehavior.h
        src/ParameterizedEnvironment.h
)

if(BUILD_BOLERO)
  #If we are building as subproject other subprojects may need to know where the includes are
  set(BOLERO_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src CACHE STRING "Include-dir for bolero.")
endif()

# Install headers into mars include directory
install(FILES ${HEADERS} DESTINATION include/${PROJECT_NAME})

# Prepare and install necessary files to support finding of the library 
# using pkg-config
configure_file(${PROJECT_NAME}.pc.in ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc DESTINATION lib/pkgconfig)
