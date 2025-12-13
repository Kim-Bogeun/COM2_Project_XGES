cmake_minimum_required(VERSION 3.22)
project(xges LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

# Default to a modern architecture, can be overridden by the user.
set(CMAKE_CUDA_ARCHITECTURES 86)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_subdirectory(lib)
add_subdirectory(src-cpp)
add_subdirectory(tests)
