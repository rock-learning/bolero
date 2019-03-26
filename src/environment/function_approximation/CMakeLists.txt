project(function_approximation)
set(PROJECT_VERSION 1.0)
set(PROJECT_DESCRIPTION "      This environment reads an data input file and searches for a behavior that approximates the data.")

include(FindPkgConfig)
find_package(lib_manager)
lib_defaults()
define_module_info()

cmake_minimum_required(VERSION 2.6)


pkg_check_modules(PKGCONFIG lib_manager
                            configmaps
                            bolero )
include_directories(${PKGCONFIG_INCLUDE_DIRS})
link_directories(${PKGCONFIG_LIBRARY_DIRS})

include_directories(
	src
)

set(SOURCES 
        src/FunctionApproximation.cpp
)

set(HEADERS
        src/FunctionApproximation.h
)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_link_libraries(${PROJECT_NAME}
                      ${PKGCONFIG_LIBRARIES}
)

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


# Install the library into the lib folder
install(TARGETS ${PROJECT_NAME} ${_INSTALL_DESTINATIONS})

# Install headers into mars include directory
install(FILES ${HEADERS} DESTINATION include/bolero/${PROJECT_NAME})

# Prepare and install necessary files to support finding of the library 
# using pkg-config
configure_file(${PROJECT_NAME}.pc.in ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc DESTINATION lib/pkgconfig)
