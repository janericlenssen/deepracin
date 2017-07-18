cmake_minimum_required (VERSION 2.8)
if (CMAKE_VERSION VERSION_LESS 3.4 AND WIN32)
  message(FATAL_ERROR "CMake 3.4 required on Windows.")
endif()

project (deepracin)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Find files
file(GLOB SOURCES src/*.c)
file(GLOB HEADERS include/*.h)
file(GLOB KERNELS Kernels/*.cl)

# Options
option (INSTALL_PYTHON_INTERFACE 
        "Install the python interface of deepRacin (Python and Numpy required)" ON) 
option (COMPILE_EXAMPLES 
        "Compile deepRacin examples (OpenCV required)" ON) 

# Include header files
include_directories(include)

# Libraries
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

find_package( OpenCL REQUIRED )

include_directories( ${OPENCL_INCLUDE_DIRS} )

if( OPENCL_HAS_CPP_BINDINGS )
	message( "OpenCL has CPP bindings. Full include is: " ${OPENCL_INCLUDE_DIRS} )
else( OPENCL_HAS_CPP_BINDINGS )
	message( "No OpenCL CPP bindings found" )
endif( OPENCL_HAS_CPP_BINDINGS )

find_package (GLIB REQUIRED)
include_directories( ${GLIB_INCLUDE_DIRS} )
message( "Glib include dir: " ${GLIB_INCLUDE_DIRS} )

# Create shared and non-shared library
add_library(${PROJECT_NAME} SHARED ${SOURCES})

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )

# Definitions
cmake_policy(SET CMP0005 NEW) # Escape preprocessor makros automatically
add_definitions(-DKERNEL_PATH="${CMAKE_INSTALL_PREFIX}/lib/deepracin/kernels/")

# Link libraries
target_link_libraries (${PROJECT_NAME} ${GLIB_LIBRARIES})
target_link_libraries (${PROJECT_NAME} ${OPENCL_LIBRARIES})

# Install library
install(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})
install(FILES ${HEADERS} DESTINATION include/${PROJECT_NAME})
install(FILES ${KERNELS} DESTINATION lib/${PROJECT_NAME}/kernels/)

# Install examples
if (COMPILE_EXAMPLES)
	add_subdirectory (examples)
endif(COMPILE_EXAMPLES)
# Install python interface
if (INSTALL_PYTHON_INTERFACE)
	add_subdirectory (python)
endif(INSTALL_PYTHON_INTERFACE)
