cmake_minimum_required(VERSION 2.6)

project(mountain_car)
set(PROJECT_VERSION 1.0)
set(PROJECT_DESCRIPTION "Mountain car learning problem.")

include(FindPkgConfig)
find_package(lib_manager)
lib_defaults()
define_module_info()

if(WIN32)
        # this fixes the error 998 from the LibManager
        SET(CMAKE_SHARED_LINKER_FLAGS "-Wl,--enable-auto-import -Wall")
        SET(CMAKE_MODULE_LINKER_FLAGS "-Wl,--enable-auto-import -Wall")
else(WIN32)
        SET(CMAKE_CXX_FLAGS "-fPIC -Wall")
endif(WIN32)

pkg_check_modules(LIB_MANAGER REQUIRED "lib_manager")
include_directories(${LIB_MANAGER_INCLUDE_DIRS})
link_directories(${LIB_MANAGER_LIBRARY_DIRS})

pkg_check_modules(BOLERO REQUIRED bolero)
include_directories(${BOLERO_INCLUDE_DIRS})

set(SOURCES src/MountainCar.cpp)
set(HEADERS src/MountainCar.h)

include_directories(src)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_link_libraries(${PROJECT_NAME}
                      ${LIB_MANAGER_LIBRARIES}
                      ${MARS_UTILS_LIBRARIES})

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

install(TARGETS ${PROJECT_NAME} ${_INSTALL_DESTINATIONS})
install(FILES ${HEADERS} DESTINATION include/bolero/${PROJECT_NAME})
configure_file(${PROJECT_NAME}.pc.in ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc DESTINATION lib/pkgconfig)
