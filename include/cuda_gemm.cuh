#ifndef CUDA_GEMM_CUH
#define CUDA_GEMM_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

// CUDA Macro for robust error checking
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
inline void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << "\n";
        std::cerr << cudaGetErrorString(err) << " " << func << "\n";
        exit(EXIT_FAILURE);
    }
}

// Tile size constants for shared memory kernel optimizations
constexpr int TILE_SIZE = 32;

namespace benchmark {

// Kernel 1: Naive CUDA GEMM (1 thread per output element C[i, j])
__global__ void gemm_cuda_naive_kernel(const float* __restrict__ A, 
                                        const float* __restrict__ B, 
                                        float* __restrict__ C, 
                                        int M, int N, int K);

// Kernel 2: Shared Memory Tiled CUDA GEMM (2D Tiling with TILE_SIZE x TILE_SIZE blocks)
__global__ void gemm_cuda_tiled_kernel(const float* __restrict__ A, 
                                        const float* __restrict__ B, 
                                        float* __restrict__ C, 
                                        int M, int N, int K);

// Kernel 3: Shared Memory Tiled + Float4 Vectorized CUDA GEMM
__global__ void gemm_cuda_vectorized_kernel(const float* __restrict__ A, 
                                             const float* __restrict__ B, 
                                             float* __restrict__ C, 
                                             int M, int N, int K);

// C++ Host Wrapper Functions
void run_cuda_naive(const float* h_A, const float* h_B, float* h_C, int M, int N, int K, float& time_ms);
void run_cuda_tiled(const float* h_A, const float* h_B, float* h_C, int M, int N, int K, float& time_ms);
void run_cuda_vectorized(const float* h_A, const float* h_B, float* h_C, int M, int N, int K, float& time_ms);

} // namespace benchmark

#endif // CUDA_GEMM_CUH
