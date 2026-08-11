cmake_minimum_required(VERSION 3.20)
project(gpu_kernel_benchmark_suite LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable Multithreading support
find_package(Threads REQUIRED)

# Include header directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Optimization flags based on compiler and architecture
if(MSVC)
    add_compile_options(/O2 /W4 /fp:fast)
else()
    add_compile_options(-O3 -Wall -Wextra -Wpedantic)
    add_compile_options(-march=native)
endif()

# Enable CUDA support conditionally if compiler supports it
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xptxas -O3")
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
else()
    message(STATUS "No CUDA compiler found. Skipping CUDA target (CPU baseline executable only).")
endif()

# CPU Baseline Executable
add_executable(cpu_benchmark
    src/cpu_baseline.cpp
    src/main_cpu.cpp
)
target_link_libraries(cpu_benchmark PRIVATE Threads::Threads)

# CUDA Benchmark Executable (if CUDA is enabled)
if(CMAKE_CUDA_COMPILER)
    add_executable(cuda_benchmark
        src/cuda_gemm.cu
        src/main_cuda.cu
        src/cpu_baseline.cpp
    )
    set_target_properties(cuda_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(cuda_benchmark PRIVATE Threads::Threads)
endif()

# Informational output
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Target Processor Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
