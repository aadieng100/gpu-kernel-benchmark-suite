#include "cuda_gemm.cuh"
#include "matrix_utils.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace benchmark;

void run_cuda_benchmarks_for_size(int size) {
    int M = size, N = size, K = size;
    std::cout << "\n========================================================\n";
    std::cout << " CUDA GPU BENCHMARK - MATRIX SIZE: " << M << " x " << N << " x " << K << "\n";
    std::cout << "========================================================\n";

    std::vector<float> h_A, h_B, h_C_ref(M * N, 0.0f), h_C_test(M * N, 0.0f);
    init_random_matrix(h_A, M, K);
    init_random_matrix(h_B, K, N);

    // Compute CPU reference for verification (for smaller matrix sizes to save time)
    if (size <= 1024) {
        gemm_cpu_cache_optimized(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
    }

    // 1. Naive CUDA Kernel
    float time_naive_ms = 0.0f;
    run_cuda_naive(h_A.data(), h_B.data(), h_C_test.data(), M, N, K, time_naive_ms);
    double gflops_naive = calculate_gflops(M, N, K, time_naive_ms);
    
    double max_err = 0.0;
    bool pass_naive = (size > 1024) || verify_matrix(h_C_test, h_C_ref, M, N, max_err);
    std::cout << "1. Naive CUDA Kernel       : " << std::fixed << std::setprecision(3) 
              << time_naive_ms << " ms | " << gflops_naive << " GFLOPS | "
              << (pass_naive ? "PASSED" : "FAILED") << "\n";

    // 2. Shared Memory Tiled CUDA Kernel
    float time_tiled_ms = 0.0f;
    run_cuda_tiled(h_A.data(), h_B.data(), h_C_test.data(), M, N, K, time_tiled_ms);
    double gflops_tiled = calculate_gflops(M, N, K, time_tiled_ms);
    
    bool pass_tiled = (size > 1024) || verify_matrix(h_C_test, h_C_ref, M, N, max_err);
    std::cout << "2. Tiled CUDA Kernel (SRAM): " << std::fixed << std::setprecision(3) 
              << time_tiled_ms << " ms | " << gflops_tiled << " GFLOPS | "
              << (pass_tiled ? "PASSED" : "FAILED") << "\n";

    // 3. Vectorized Float4 + 2D Tiled CUDA Kernel
    float time_vec_ms = 0.0f;
    run_cuda_vectorized(h_A.data(), h_B.data(), h_C_test.data(), M, N, K, time_vec_ms);
    double gflops_vec = calculate_gflops(M, N, K, time_vec_ms);
    
    bool pass_vec = (size > 1024) || verify_matrix(h_C_test, h_C_ref, M, N, max_err);
    std::cout << "3. Vectorized 2D-Tiled CUDA: " << std::fixed << std::setprecision(3) 
              << time_vec_ms << " ms | " << gflops_vec << " GFLOPS | "
              << (pass_vec ? "PASSED" : "FAILED") << "\n";

    // Summary Table
    std::cout << "\n+------------------------------------+----------------+--------------+------------------+------------+\n";
    std::cout << "| CUDA Implementation                | Time (ms)      | GFLOPS       | Speedup vs Naive | Verified   |\n";
    std::cout << "+------------------------------------+----------------+--------------+------------------+------------+\n";
    std::cout << "| 1. Naive CUDA                      | " << std::setw(14) << time_naive_ms << " | " << std::setw(12) << gflops_naive << " | " << std::setw(16) << "1.00x" << " | " << std::setw(10) << (pass_naive ? "PASSED" : "FAILED") << " |\n";
    std::cout << "| 2. Shared Memory Tiled (32x32)     | " << std::setw(14) << time_tiled_ms << " | " << std::setw(12) << gflops_tiled << " | " << std::setw(15) << (time_naive_ms / time_tiled_ms) << "x | " << std::setw(10) << (pass_tiled ? "PASSED" : "FAILED") << " |\n";
    std::cout << "| 3. Vectorized float4 + 2D Tiling   | " << std::setw(14) << time_vec_ms << " | " << std::setw(12) << gflops_vec << " | " << std::setw(15) << (time_naive_ms / time_vec_ms) << "x | " << std::setw(10) << (pass_vec ? "PASSED" : "FAILED") << " |\n";
    std::cout << "+------------------------------------+----------------+--------------+------------------+------------+\n";
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "========================================================================\n";
    std::cout << " GPU KERNEL BENCHMARK SUITE - PHASE 2: CUDA GPU KERNELS\n";
    std::cout << " GPU Device           : " << prop.name << "\n";
    std::cout << " SM Count             : " << prop.multiProcessorCount << "\n";
    std::cout << " Shared Memory / Block: " << (prop.sharedMemPerBlock / 1024) << " KB\n";
    std::cout << " Global Memory        : " << (prop.totalGlobalMem / (1024 * 1024 * 1024)) << " GB\n";
    std::cout << "========================================================================\n";

    std::vector<int> test_sizes = {512, 1024, 2048, 4096};
    for (int size : test_sizes) {
        run_cuda_benchmarks_for_size(size);
    }

    std::cout << "\nPhase 2 CUDA GPU Benchmark completed successfully.\n";
    return 0;
}
