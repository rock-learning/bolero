project(test_py_loadable_behavior)
cmake_minimum_required(VERSION 2.6)

set(PROJECT_VERSION 1.0)
set(PROJECT_DESCRIPTION "")

include(FindPkgConfig)
find_package(lib_manager)
lib_defaults()
define_module_info()


# Get linker and compiler flags from pkg-config
pkg_check_modules(MARS_UTILS REQUIRED mars_utils)
include_directories(${MARS_UTILS_INCLUDE_DIRS})
link_directories(${MARS_UTILS_LIBRARY_DIRS})
add_definitions(${MARS_UTILS_CFLAGS_OTHER})  #flags excluding the ones with -I


pkg_check_modules(BOLERO REQUIRED bolero)
include_directories(${BOLERO_INCLUDE_DIRS}) 
include_directories(${BL_LOADER_INCLUDE_DIRS})   
include_directories(src)

set(SOURCES 
        src/main.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME}
                      ${MARS_UTILS_LIBRARIES}
)


add_dependencies(${PROJECT_NAME} BOLERO)
target_link_libraries(${PROJECT_NAME} ${BL_LOADER_LIBRARIES}) 



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

configure_file(config/init.yaml.in ${CMAKE_CURRENT_BINARY_DIR}/init.yaml @ONLY)
configure_file(config/config.yaml.in ${CMAKE_CURRENT_BINARY_DIR}/config.yaml @ONLY)
configure_file(python/test_behavior.py ${CMAKE_CURRENT_BINARY_DIR}/test_behavior.py @ONLY)

enable_testing()
add_test(test_py_loadable_behavior test_py_loadable_behavior)

