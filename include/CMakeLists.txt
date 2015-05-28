cmake_minimum_required(VERSION 2.6)

project(bolero)
set(PROJECT_VERSION 0.1)
set(PROJECT_DESCRIPTION "Behavior Optimization and Learning for Robots")

set(HEADERS
  src/Behavior.h
  src/BehaviorSearch.h
  src/Optimizer.h
  src/ContextualEnvironment.h
  src/Environment.h
  src/LoadableBehavior.h
  src/ParameterizedEnvironment.h
)

install(FILES ${HEADERS} DESTINATION include/${PROJECT_NAME})
configure_file(${PROJECT_NAME}.pc.in ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc DESTINATION lib/pkgconfig)
