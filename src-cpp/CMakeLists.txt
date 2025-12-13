set(XGES_LIB_SOURCES
    PDAG.cpp
    PDAG.h
    ScorerInterface.h
    Operators.h
    Operators.cpp
    set_ops.h
    set_ops.cpp
    EdgeQueueSet.cpp
    EdgeQueueSet.h
    XGES.cpp
    XGES.h
    XGESBatched.cpp
    XGESBatched.h
    BICScorer.cpp
    BICScorer.h
    BICScorerGPU.cu
    BICScorerGPU_Optimized.cu
    BICScorerGPU_Optimized.h
    BICScorerGPU_cuBLAS.cu
    BICScorerGPU_cuBLAS.h
    CircularBuffer.h
    utils.h
)

# Benchmark-only library (without XGES.cpp which requires spdlog)
set(BENCHMARK_LIB_SOURCES
    BICScorer.cpp
    BICScorer.h
    BICScorerGPU.cu
    BICScorerGPU_Optimized.cu
    BICScorerGPU_Optimized.h
    BICScorerGPU_cuBLAS.cu
    BICScorerGPU_cuBLAS.h
    set_ops.h
    set_ops.cpp
    utils.h
)

add_library(benchmark_lib STATIC ${BENCHMARK_LIB_SOURCES})

add_library(xges_lib STATIC ${XGES_LIB_SOURCES})

# Find and include Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_include_directories(xges_lib PRIVATE ${EIGEN3_INCLUDE_DIR})
target_include_directories(xges_lib PRIVATE ${CMAKE_SOURCE_DIR}/lib/spdlog/include)
target_include_directories(benchmark_lib PRIVATE ${EIGEN3_INCLUDE_DIR})
target_include_directories(benchmark_lib PRIVATE ${CMAKE_SOURCE_DIR}/lib/spdlog/include)

# --- Main executable
set(BINARY ${CMAKE_PROJECT_NAME})
add_executable(${BINARY} main.cpp)
target_link_libraries(${BINARY} PRIVATE xges_lib)
target_include_directories(${BINARY} PRIVATE ${EIGEN3_INCLUDE_DIR})
target_include_directories(${BINARY} PRIVATE ${CMAKE_SOURCE_DIR}/lib/spdlog/include)


# --- Link libraries to the static lib
target_link_libraries(xges_lib PRIVATE cnpy)
target_link_libraries(xges_lib PRIVATE spdlog::spdlog)

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(xges_lib PRIVATE ${Boost_LIBRARIES})
target_link_libraries(benchmark_lib PRIVATE ${Boost_LIBRARIES})

find_package(BLAS REQUIRED)
include_directories(${BLAS_INCLUDE_DIRS})
target_link_libraries(xges_lib PRIVATE ${BLAS_LIBRARIES})
target_link_libraries(benchmark_lib PRIVATE ${BLAS_LIBRARIES})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(xges_lib PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusolver_LIBRARY})
target_link_libraries(benchmark_lib PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusolver_LIBRARY})
