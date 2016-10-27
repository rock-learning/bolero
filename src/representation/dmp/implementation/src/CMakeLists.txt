include_directories(${EIGEN3_INCLUDE_DIRS} .)
add_library(dmp SHARED Dmp.cpp)

install (TARGETS dmp DESTINATION lib)
install (FILES ${headers} DESTINATION include/${PROJECT_NAME})
