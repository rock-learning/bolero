cmake_minimum_required(VERSION 2.8)

project(mars_environment)
set(PROJECT_VERSION 1.0)
set(PROJECT_DESCRIPTION "Common MARS environment")

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

pkg_check_modules(BOLERO REQUIRED bolero)
include_directories(${BOLERO_INCLUDE_DIRS})

pkg_check_modules(LIB_MANAGER "lib_manager")
include_directories(${LIB_MANAGER_INCLUDE_DIRS})
link_directories(${LIB_MANAGER_LIBRARY_DIRS})

pkg_check_modules(CFG_MANAGER "cfg_manager")
include_directories(${CFG_MANAGER_INCLUDE_DIRS})
link_directories(${CFG_MANAGER_LIBRARY_DIRS})

pkg_check_modules(DATA_BROKER "data_broker")
include_directories(${DATA_BROKER_INCLUDE_DIRS})
link_directories(${DATA_BROKER_LIBRARY_DIRS})

pkg_check_modules(CONFIGMAPS "configmaps")
include_directories(${CONFIGMAPS_INCLUDE_DIRS})
link_directories(${CONFIGMAPS_LIBRARY_DIRS})

pkg_check_modules(MARS_APP "mars_app")
include_directories(${MARS_APP_INCLUDE_DIRS})
link_directories(${MARS_APP_LIBRARY_DIRS})

pkg_check_modules(MARS_UTILS "mars_utils")
include_directories(${MARS_UTILS_INCLUDE_DIRS})
link_directories(${MARS_UTILS_LIBRARY_DIRS})

pkg_check_modules(MARS_INTERFACES "mars_interfaces")
include_directories(${MARS_INTERFACES_INCLUDE_DIRS})
link_directories(${MARS_INTERFACES_LIBRARY_DIRS})

include_directories(src)

set(SOURCES 
    src/MARSEnvironment.cpp
    src/MARSEnvironmentHelper.cpp
    src/MARSEnvPlugin.cpp
    src/MARSThread.cpp
    src/MARSReceiver.cpp
)

set(HEADERS
    src/MARSEnvironment.h
    src/MARSEnvironmentHelper.h
    src/MARSEnvPlugin.h
    src/MARSReceiver.h
)

# without main_gui we don't need qt
pkg_check_modules(MAIN_GUI "main_gui")
if(MAIN_GUI_FOUND)
  setup_qt()
else(MAIN_GUI_FOUND)
  add_definitions("-DNO_GUI")
endif(MAIN_GUI_FOUND)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

if (${USE_QT5})
  qt5_use_modules(${PROJECT_NAME} Widgets)
endif (${USE_QT5})

target_link_libraries(${PROJECT_NAME}
                      ${LIB_MANAGER_LIBRARIES}
                      ${CFG_MANAGER_LIBRARIES}
                      ${DATA_BROKER_LIBRARIES}
                      ${CONFIGMAPS_LIBRARIES}
                      ${MARS_APP_LIBRARIES}
                      ${MARS_UTILS_LIBRARIES}
                      ${MARS_INTERFACES_LIBRARIES}
                      ${QT_LIBRARIES}
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


IF (WIN32)
  SET (POSTLIB ".dll")
  SET (PRELIB "lib")
  SET (LIBPATH "bin")
ELSE (WIN32)
  IF (APPLE)
    SET (POSTLIB ".dylib")
    SET (PRELIB "lib")
    SET (LIBPATH "lib")
  ELSE (APPLE)
    SET (POSTLIB ".so")
    SET (PRELIB "lib")
    SET (LIBPATH "lib")
  ENDIF (APPLE)
ENDIF (WIN32)

# Install the library into the lib folder
install(TARGETS ${PROJECT_NAME} ${_INSTALL_DESTINATIONS})

# Install headers into mars include directory
install(FILES ${HEADERS} DESTINATION include/bolero/${PROJECT_NAME})

# Prepare and install necessary files to support finding of the library 
# using pkg-config
configure_file(${PROJECT_NAME}.pc.in ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc DESTINATION lib/pkgconfig)
#install(DIRECTORY input_data DESTINATION configuration/learning_controller_default)
