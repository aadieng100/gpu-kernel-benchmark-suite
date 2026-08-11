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

# CPU Baseline Executable
add_executable(cpu_benchmark
    src/cpu_baseline.cpp
    src/main_cpu.cpp
)

target_link_libraries(cpu_benchmark PRIVATE Threads::Threads)

# Informational output
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Target Processor Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
