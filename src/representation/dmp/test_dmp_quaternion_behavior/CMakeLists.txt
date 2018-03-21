project(dmp_quaternion_behavior_test)
cmake_minimum_required(VERSION 2.6)

set(PROJECT_VERSION 1.0)
set(PROJECT_DESCRIPTION "")

include(FindPkgConfig)
find_package(lib_manager)
lib_defaults()
define_module_info()

include_directories(
    ${BL_LOADER_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${YAMLCPP_INCLUDE_DIRS}
    ${BOLERO_INCLUDE_DIRS})
link_directories(${BL_LOADER_LIBRARY_DIRS})
link_directories(${YAMLCPP_LIBRARY_DIRS})
add_definitions(${BL_LOADER_CFLAGS_OTHER})

include_directories(src)
set(SOURCES src/main.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${BL_LOADER_LIBRARIES} quaternion_dmp_cpp)

if(WIN32)
  set(LIB_INSTALL_DIR bin) # .dll are in PATH, like executables
else(WIN32)
  set(LIB_INSTALL_DIR lib)
endif(WIN32)


set(_INSTALL_DESTINATIONS
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION ${LIB_INSTALL_DIR}
	ARCHIVE DESTINATION lib
)

IF (WIN32)
  SET (POSTLIB ".dll")
  SET (PRELIB "lib")
  SET (LIBPATH "bin")
  SET (LIBRARY_PATH_VAR "LD_LIBRARY_PATH")
ELSE (WIN32)
  IF (APPLE)
    SET (POSTLIB ".dylib")
    SET (PRELIB "lib")
    SET (LIBPATH "lib")
    SET (LIBRARY_PATH_VAR "DYLD_LIBRARY_PATH")
  ELSE (APPLE)
    SET (POSTLIB ".so")
    SET (PRELIB "lib")
    SET (LIBPATH "lib")
    SET (LIBRARY_PATH_VAR "LD_LIBRARY_PATH")
  ENDIF (APPLE)
ENDIF (WIN32)

# Install the library into the lib folder
install(TARGETS ${PROJECT_NAME} ${_INSTALL_DESTINATIONS})

# Install headers into mars include directory
#install(FILES ${HEADERS} DESTINATION include/${PROJECT_NAME})

# Prepare and install necessary files to support finding of the library 
# using pkg-config
#configure_file(${PROJECT_NAME}.pc.in ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc @ONLY)
#install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc DESTINATION lib/pkgconfig)

configure_file(configuration/load_libraries.in ${CMAKE_CURRENT_BINARY_DIR}/load_libraries.txt @ONLY)
configure_file(configuration/model.yaml ${CMAKE_CURRENT_BINARY_DIR}/model.yaml @ONLY)
configure_file(configuration/config.yaml ${CMAKE_CURRENT_BINARY_DIR}/config.yaml @ONLY)

enable_testing()
add_test(${PROJECT_NAME} ${PROJECT_NAME})



