cmake_minimum_required(VERSION 2.6)


include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${YAMLCPP_INCLUDE_DIR}
    ${BOLERO_INCLUDE_DIRS})

set(sources CanonicalSystem.cpp
            ForcingTerm.cpp
            TransformationSystem.cpp
            Dmp.cpp
            FifthOrderPolynomial.cpp
            RbfFunctionApproximator.cpp
            DMPConfig.cpp
            DmpBehavior.cpp
            DMPModel.cpp
            )
set(headers FifthOrderPolynomial.h
            RbfFunctionApproximator.h
            CanonicalSystem.h
            ForcingTerm.h
            TransformationSystem.h
            Dmp.h
            DMPModel.h
            DMPConfig.h
            DMPWrapper.h
            DmpBehavior.h
            EigenHelpers.h
            )

set(quaternion_headers RbfFunctionApproximator.h
                       CanonicalSystem.h
                       ForcingTerm.h
                       DMPModel.h
                       QuaternionDmp.h
                       QuaternionTransformationSystem.h
                       EigenHelpers.h
                       QuaternionDmpConfig.h
                       QuaternionDmpModel.h)

set(quaternion_sources CanonicalSystem.cpp
                       ForcingTerm.cpp
                       RbfFunctionApproximator.cpp
                       DMPModel.cpp
                       QuaternionDmp.cpp
                       QuaternionTransformationSystem.cpp
                       QuaternionDmpConfig.cpp)

set(rigid_body_headers RbfFunctionApproximator.h
                       CanonicalSystem.h
                       ForcingTerm.h
                       DMPModel.h
                       QuaternionDmp.h
                       QuaternionTransformationSystem.h
                       EigenHelpers.h
                       QuaternionDmpConfig.h
                       QuaternionDmpModel.h
                       DMPConfig.h
                       TransformationSystem.h
                       Dmp.h
                       RigidBodyDmp.h
                       RigidBodyDmpConfig.h
                       DmpBehavior.h
                       FifthOrderPolynomial.h)

set(rigid_body_sources CanonicalSystem.cpp
                       ForcingTerm.cpp
                       RbfFunctionApproximator.cpp
                       DMPModel.cpp
                       QuaternionDmp.cpp
                       QuaternionTransformationSystem.cpp
                       QuaternionDmpConfig.cpp
                       DMPConfig.cpp
                       TransformationSystem.cpp
                       Dmp.cpp
                       RigidBodyDmp.cpp
                       RigidBodyDmpConfig.cpp
                       DmpBehavior.cpp
                       FifthOrderPolynomial.cpp)




add_library(dmp_cpp SHARED ${sources} ${headers})
target_link_libraries(dmp_cpp ${YAMLCPP_LIBRARY})
#BUILD_DMP_BEHAVIOR_STANDALONE enables the behavior macros for the DmpBehavior
set_target_properties(dmp_cpp PROPERTIES COMPILE_FLAGS "-DBUILD_DMP_BEHAVIOR_STANDALONE")

add_library(quaternion_dmp_cpp SHARED ${quaternion_sources} ${quaternion_headers})
target_link_libraries(quaternion_dmp_cpp ${YAMLCPP_LIBRARY})
# BUILD_QUATERNION_STANDALONE enables the behavior macros for the QuaternionDMP
set_target_properties(quaternion_dmp_cpp PROPERTIES COMPILE_FLAGS "-DBUILD_QUATERNION_STANDALONE")

add_library(rigid_body_dmp_cpp SHARED ${rigid_body_sources} ${rigid_body_headers})
target_link_libraries(rigid_body_dmp_cpp ${YAMLCPP_LIBRARY})

configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/dmp_cpp.pc.in"
  "${PROJECT_BINARY_DIR}/dmp_cpp.pc" @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/dmp_cpp.pc DESTINATION lib/pkgconfig)

configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/quaternion_dmp_cpp.pc.in"
    "${PROJECT_BINARY_DIR}/quaternion_dmp_cpp.pc" @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/quaternion_dmp_cpp.pc DESTINATION lib/pkgconfig)

configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/rigid_body_dmp_cpp.pc.in"
    "${PROJECT_BINARY_DIR}/rigid_body_dmp_cpp.pc" @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/rigid_body_dmp_cpp.pc DESTINATION lib/pkgconfig)

install (TARGETS dmp_cpp DESTINATION lib)
install (TARGETS quaternion_dmp_cpp DESTINATION lib)
install (TARGETS rigid_body_dmp_cpp DESTINATION lib)
install (FILES ${headers} DESTINATION include/${PROJECT_NAME})
install (FILES ${quaternion_headers} DESTINATION include/${PROJECT_NAME})
install (FILES ${rigid_body_headers} DESTINATION include/${PROJECT_NAME})
