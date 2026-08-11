#include "matrix_utils.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <string>

using namespace benchmark;

void run_benchmark_for_size(int size) {
    int M = size, N = size, K = size;
    std::cout << "\n========================================================\n";
    std::cout << " BENCHMARKING MATRIX SIZE: " << M << " x " << N << " x " << K << "\n";
    std::cout << "========================================================\n";

    std::vector<float> A, B, C_ref(M * N, 0.0f), C_test(M * N, 0.0f);
    init_random_matrix(A, M, K);
    init_random_matrix(B, K, N);

    // Warm-up & Naive CPU Benchmark
    std::cout << "Running 1. Naive CPU GEMM (i-j-k)... " << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    gemm_cpu_naive(A.data(), B.data(), C_ref.data(), M, N, K);
    auto end = std::chrono::high_resolution_clock::now();
    double time_naive_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gflops_naive = calculate_gflops(M, N, K, time_naive_ms);
    std::cout << "Done (" << std::fixed << std::setprecision(2) << time_naive_ms << " ms, " << gflops_naive << " GFLOPS)\n";

    // Cache-Optimized CPU Benchmark
    std::cout << "Running 2. Cache-Optimized CPU GEMM (i-k-j)... " << std::flush;
    start = std::chrono::high_resolution_clock::now();
    gemm_cpu_cache_optimized(A.data(), B.data(), C_test.data(), M, N, K);
    end = std::chrono::high_resolution_clock::now();
    double time_cache_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gflops_cache = calculate_gflops(M, N, K, time_cache_ms);
    
    double max_err = 0.0;
    bool pass_cache = verify_matrix(C_test, C_ref, M, N, max_err);
    std::cout << "Done (" << std::fixed << std::setprecision(2) << time_cache_ms << " ms, " 
              << gflops_cache << " GFLOPS, Valid: " << (pass_cache ? "PASSED" : "FAILED") << ")\n";

    // Multi-threaded CPU Benchmark
    unsigned int hw_threads = std::thread::hardware_concurrency();
    std::cout << "Running 3. Multi-Threaded CPU GEMM (" << hw_threads << " threads)... " << std::flush;
    start = std::chrono::high_resolution_clock::now();
    gemm_cpu_multithreaded(A.data(), B.data(), C_test.data(), M, N, K, hw_threads);
    end = std::chrono::high_resolution_clock::now();
    double time_mt_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gflops_mt = calculate_gflops(M, N, K, time_mt_ms);
    
    bool pass_mt = verify_matrix(C_test, C_ref, M, N, max_err);
    std::cout << "Done (" << std::fixed << std::setprecision(2) << time_mt_ms << " ms, " 
              << gflops_mt << " GFLOPS, Valid: " << (pass_mt ? "PASSED" : "FAILED") << ")\n";

    // Summary Table
    std::cout << "\n+------------------------------------+----------------+--------------+------------------+------------+\n";
    std::cout << "| Implementation                     | Time (ms)      | GFLOPS       | Speedup vs Naive | Verified   |\n";
    std::cout << "+------------------------------------+----------------+--------------+------------------+------------+\n";
    std::cout << "| 1. CPU Naive (i-j-k)               | " << std::setw(14) << time_naive_ms << " | " << std::setw(12) << gflops_naive << " | " << std::setw(16) << "1.00x" << " | " << std::setw(10) << "REF" << " |\n";
    std::cout << "| 2. CPU Cache-Optimized (i-k-j)     | " << std::setw(14) << time_cache_ms << " | " << std::setw(12) << gflops_cache << " | " << std::setw(15) << (time_naive_ms / time_cache_ms) << "x | " << std::setw(10) << (pass_cache ? "PASSED" : "FAILED") << " |\n";
    std::cout << "| 3. CPU Multi-Threaded (" << hw_threads << " cores)     | " << std::setw(14) << time_mt_ms << " | " << std::setw(12) << gflops_mt << " | " << std::setw(15) << (time_naive_ms / time_mt_ms) << "x | " << std::setw(10) << (pass_mt ? "PASSED" : "FAILED") << " |\n";
    std::cout << "+------------------------------------+----------------+--------------+------------------+------------+\n";
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << " GPU KERNEL BENCHMARK SUITE - PHASE 1: CPU BASELINE COMPARISON\n";
    std::cout << " Hardware Threads Available: " << std::thread::hardware_concurrency() << "\n";
    std::cout << "========================================================================\n";

    std::vector<int> test_sizes = {256, 512, 1024};
    
    for (int size : test_sizes) {
        run_benchmark_for_size(size);
    }

    std::cout << "\nPhase 1 CPU Baseline Benchmark completed successfully.\n";
    return 0;
}
