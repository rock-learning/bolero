pkg_check_modules(CATCH REQUIRED catch)

include_directories(
    ${LIB_MANAGER_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${YAMLCPP_INCLUDE_DIRS}
    ${BOLERO_INCLUDE_DIRS}
    ${CATCH_INCLUDE_DIRS})
link_directories(${LIB_MANAGER_LIBRARY_DIRS})
link_directories(${YAMLCPP_LIBRARY_DIRS})

set(dmp_sources
    test_dmp_config.cpp
    test_dmp_model.cpp
    test_main.cpp)

set(dmp_quaternion_behavior_sources
    test_main.cpp
    test_quaternion_dmp.cpp
    test_quaternion_dmp_config.cpp)

set(dmp_rigid_body_sources
    test_main.cpp
    test_rigid_body_dmp_config.cpp
    test_rigid_body_dmp.cpp)


file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/model.yaml DESTINATION .)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/config.yaml DESTINATION .)

add_executable (dmp_cpp_test ${dmp_sources})
target_link_libraries(dmp_cpp_test dmp_cpp ${LIB_MANAGER_LIBRARIES})

# Behaviors need to be tested seperatly because multiple behaviors cannot exist
# in the same library
add_executable(dmp_cpp_behavior_test
    test_main.cpp test_dmp_behavior.cpp)
target_link_libraries(dmp_cpp_behavior_test
    dmp_cpp ${LIB_MANAGER_LIBRARIES})

add_executable(dmp_cpp_quaternion_behavior_test
    ${dmp_quaternion_behavior_sources})
target_link_libraries(dmp_cpp_quaternion_behavior_test
    quaternion_dmp_cpp ${LIB_MANAGER_LIBRARIES})

add_executable(dmp_cpp_rigid_body_test ${dmp_rigid_body_sources})
target_link_libraries(dmp_cpp_rigid_body_test
    rigid_body_dmp_cpp ${LIB_MANAGER_LIBRARIES})
set_target_properties(dmp_cpp_rigid_body_test PROPERTIES COMPILE_FLAGS "-std=c++0x")


enable_testing()
add_test(dmp_test dmp_cpp_test)
add_test(dmp_behavior_test dmp_cpp_behavior_test)
add_test(dmp_quaternion_test dmp_cpp_quaternion_behavior_test)
add_test(dmp_rigid_body_test dmp_cpp_rigid_body_test)

