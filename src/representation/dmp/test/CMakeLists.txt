include_directories(
    ${LIB_MANAGER_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${YAMLCPP_INCLUDE_DIRS}
    ${BOLERO_INCLUDE_DIRS})
link_directories(${LIB_MANAGER_LIBRARY_DIRS})

set(dmp_sources
    test_canonical_system.cpp
    test_dmp.cpp
    test_dmp_config.cpp
    test_dmp_model.cpp
    test_dmp_wrapper.cpp
    test_fifth_order_polynomial.cpp
    test_forcing_term.cpp
    test_function_approximator.cpp
    test_main.cpp
    test_transformation_system.cpp)

set(dmp_quaternion_behavior_sources
    test_main.cpp
    test_quaternion_dmp.cpp
    test_quaternion_transformation_system.cpp
    test_quaternion_dmp_config.cpp)

set(dmp_rigid_body_sources
    test_main.cpp
    test_rigid_body_dmp_config.cpp
    test_rigid_body_dmp.cpp)

add_executable (dmp_cpp_test ${dmp_sources} catch.hpp)
target_link_libraries(dmp_cpp_test dmp_cpp ${LIB_MANAGER_LIBRARIES})

# Behaviors need to be tested seperatly because multiple behaviors cannot exist
# in the same library
add_executable(dmp_cpp_behavior_test
    test_main.cpp test_dmp_behavior.cpp catch.hpp)
target_link_libraries(dmp_cpp_behavior_test
    dmp_cpp ${LIB_MANAGER_LIBRARIES})

add_executable(dmp_cpp_quaternion_behavior_test
    ${dmp_quaternion_behavior_sources} catch.hpp)
target_link_libraries(dmp_cpp_quaternion_behavior_test
    quaternion_dmp_cpp ${LIB_MANAGER_LIBRARIES})

add_executable(dmp_cpp_rigid_body_test ${dmp_rigid_body_sources} catch.hpp)
target_link_libraries(dmp_cpp_rigid_body_test
    rigid_body_dmp_cpp ${LIB_MANAGER_LIBRARIES})
set_target_properties(dmp_cpp_rigid_body_test PROPERTIES COMPILE_FLAGS "-std=c++0x")


enable_testing()
add_test(dmp_test dmp_cpp_test)
add_test(dmp_behavior_test dmp_cpp_behavior_test)
add_test(dmp_quaternion_test dmp_cpp_quaternion_behavior_test)
add_test(dmp_rigid_body_test dmp_cpp_rigid_body_test)

