include_directories(${EIGEN3_INCLUDE_DIRS} .)
add_library(dmp SHARED Dmp.cpp)

install (TARGETS dmp DESTINATION lib)
install (FILES Dmp.h DESTINATION include/${PROJECT_NAME})
