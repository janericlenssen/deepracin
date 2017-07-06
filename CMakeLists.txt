cmake_minimum_required (VERSION 2.4.0)
project (deepracin)

# Find files
file(GLOB SOURCES src/*.c)
file(GLOB HEADERS include/*.h)
file(GLOB KERNELS Kernels/*.cl)

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

# Create shared library
add_library(${PROJECT_NAME} SHARED ${SOURCES})

# Definitions
#add_definitions(-DKERNEL_PATH=""lib/deepracin/kernels/"")
add_definitions(-DKERNEL_PATH="\\"${CMAKE_INSTALL_PREFIX}/lib/deepracin/kernels/\\"")

# Link libraries
target_link_libraries (${PROJECT_NAME} ${GLIB_LIBRARIES})
target_link_libraries (${PROJECT_NAME} ${OPENCL_LIBRARIES})

# Install library
install(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})
install(FILES ${HEADERS} DESTINATION include/${PROJECT_NAME})
install(FILES ${KERNELS} DESTINATION lib/${PROJECT_NAME}/kernels/)

add_subdirectory (examples)
add_subdirectory (python)

