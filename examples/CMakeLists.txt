# Find libraries
find_package (GLIB REQUIRED)
find_package (OpenCV REQUIRED)

# Include header files
include_directories( ${GLIB_INCLUDE_DIRS} )
include_directories( include/deepracin/ )
include_directories( ${OpenCV_INCLUDE_DIR} )

# Create examples
add_executable (vgg16_example vgg16/vgg_deployed.cpp)
add_executable (squeezenet_example squeezenet-v1.1/squeezenet_deployed.cpp)

# Link libraries
target_link_libraries (vgg16_example deepracin)
target_link_libraries (vgg16_example ${OpenCV_LIBRARIES})
target_link_libraries (squeezenet_example deepracin)
target_link_libraries (squeezenet_example ${OpenCV_LIBRARIES})


# Create model folder
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/build/deepRacinModels)

# Copy example images
configure_file(vgg16/synset.txt ${CMAKE_BINARY_DIR}/examples COPYONLY)
configure_file(vgg16/tiger.png ${CMAKE_BINARY_DIR}/examples COPYONLY)
configure_file(vgg16/puzzle.png ${CMAKE_BINARY_DIR}/examples COPYONLY)

# Copy Squeezenet-model
file(COPY squeezenet-v1.1/squeezenet_whole DESTINATION ${CMAKE_BINARY_DIR}/deepRacinModels)
