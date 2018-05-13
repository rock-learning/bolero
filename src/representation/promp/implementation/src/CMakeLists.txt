include_directories(${EIGEN3_INCLUDE_DIRS} .)

set(HDRS BasisFunctions.h Trajectory.h promp.h)
set(SRCS BasisFunctions.cpp Trajectory.cpp promp.cpp)


add_library(promp SHARED ${SRCS} )

                         
install (TARGETS promp DESTINATION lib)
install (FILES ${HDRS} DESTINATION include/${PROJECT_NAME})
