cmake_minimum_required(VERSION 3.10)

# Project name
project(GreyscaleCUDA LANGUAGES CXX CUDA)

# Find CUDA
find_package(CUDA REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories ("/usr/include/opencv4/")

# Set the CUDA architecture 
set(CUDA_ARCHITECTURES "60")

# Include directories
include_directories(${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

# Add the executable
add_executable(app main.cu ./code/basic.cu ./code/gaussian.cu ./code/sobel.cu)

# Link libraries
target_link_libraries(app ${CUDA_LIBRARIES} ${OpenCV_LIBS} dl rt pthread)

# Set the CUDA properties
set_target_properties(app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
)