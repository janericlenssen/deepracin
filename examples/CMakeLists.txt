# Find libraries
find_package (GLIB REQUIRED)
find_package(PNG REQUIRED)

# Include header files
include_directories( ${GLIB_INCLUDE_DIRS} )
include_directories( include/deepracin/ )
include_directories(${PNG_INCLUDE_DIR})

# Create examples
add_executable (vgg16_example vgg16/vgg_deployed.c)
add_executable (squeezenet_example squeezenet-v1.1/squeezenet_deployed.c)

# Link libraries
target_link_libraries (vgg16_example deepracin)
target_link_libraries (vgg16_example ${PNG_LIBRARY})
target_link_libraries (vgg16_example -no-pie)
target_link_libraries (squeezenet_example deepracin)
target_link_libraries (squeezenet_example ${PNG_LIBRARY})
target_link_libraries (squeezenet_example -no-pie)


# Create model folder
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/build/deepRacinModels)

# Copy example images
configure_file(vgg16/synset.txt ${CMAKE_BINARY_DIR}/examples COPYONLY)
configure_file(vgg16/tiger.png ${CMAKE_BINARY_DIR}/examples COPYONLY)
configure_file(vgg16/puzzle.png ${CMAKE_BINARY_DIR}/examples COPYONLY)

# Copy Squeezenet-model
file(COPY squeezenet-v1.1/squeezenet_whole DESTINATION ${CMAKE_BINARY_DIR}/deepRacinModels)
