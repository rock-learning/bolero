cmake_minimum_required(VERSION 2.6)

include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${YAMLCPP_INCLUDE_DIRS}
    ${LIB_MANAGER_INCLUDE_DIRS}
    ${BOLERO_INCLUDE_DIRS}
    ../implementation/src)

link_directories(${YAMLCPP_LIBRARY_DIRS})
add_definitions(${YAMLCPP_CFLAGS_OTHER})  #flags excluding the ones with -I


set(sources DmpBehavior.cpp
            DMPConfig.cpp
            DMPModel.cpp
            )
set(headers DmpBehavior.h
            DMPModel.h
            DMPConfig.h
            EigenHelpers.h
            )

set(quaternion_headers DMPModel.h
                       QuaternionDmp.h
                       EigenHelpers.h
                       QuaternionDmpConfig.h
                       QuaternionDmpModel.h)

set(quaternion_sources DMPModel.cpp
                       QuaternionDmp.cpp
                       QuaternionDmpConfig.cpp)

set(rigid_body_headers DMPModel.h
                       QuaternionDmp.h
                       EigenHelpers.h
                       QuaternionDmpConfig.h
                       QuaternionDmpModel.h
                       DMPConfig.h
                       RigidBodyDmp.h
                       RigidBodyDmpConfig.h
                       DmpBehavior.h)

set(rigid_body_sources DMPModel.cpp
                       QuaternionDmp.cpp
                       QuaternionDmpConfig.cpp
                       DMPConfig.cpp
                       RigidBodyDmp.cpp
                       RigidBodyDmpConfig.cpp
                       DmpBehavior.cpp)


add_library(dmp_cpp SHARED ${sources} ${headers})
target_link_libraries(dmp_cpp dmp ${YAMLCPP_LIBRARIES})
#BUILD_DMP_BEHAVIOR_STANDALONE enables the behavior macros for the DmpBehavior
set_target_properties(dmp_cpp PROPERTIES COMPILE_FLAGS "-DBUILD_DMP_BEHAVIOR_STANDALONE")

add_library(quaternion_dmp_cpp SHARED ${quaternion_sources} ${quaternion_headers})
target_link_libraries(quaternion_dmp_cpp dmp ${YAMLCPP_LIBRARIES})
# BUILD_QUATERNION_STANDALONE enables the behavior macros for the QuaternionDMP
set_target_properties(quaternion_dmp_cpp PROPERTIES COMPILE_FLAGS "-DBUILD_QUATERNION_STANDALONE")

add_library(rigid_body_dmp_cpp SHARED ${rigid_body_sources} ${rigid_body_headers})
target_link_libraries(rigid_body_dmp_cpp dmp ${YAMLCPP_LIBRARIES})

install (TARGETS dmp_cpp DESTINATION lib)
install (TARGETS quaternion_dmp_cpp DESTINATION lib)
install (TARGETS rigid_body_dmp_cpp DESTINATION lib)
install (FILES ${headers} DESTINATION include/${PROJECT_NAME})
install (FILES ${quaternion_headers} DESTINATION include/${PROJECT_NAME})
install (FILES ${rigid_body_headers} DESTINATION include/${PROJECT_NAME})
