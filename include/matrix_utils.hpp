#ifndef MATRIX_UTILS_HPP
#define MATRIX_UTILS_HPP

#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <cmath>
#include <string>
#include <algorithm>
#include <thread>

namespace benchmark {

// Structure to store benchmark result metrics
struct BenchmarkResult {
    std::string name;
    double execution_time_ms;
    double gflops;
    bool verified;
    double max_error;
};

// Initialize matrix with random float values in range [min_val, max_val]
inline void init_random_matrix(std::vector<float>& mat, int rows, int cols, float min_val = -1.0f, float max_val = 1.0f) {
    mat.resize(rows * cols);
    std::mt19937 rng(42); // Fixed seed for reproducible benchmarks across implementations
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (auto& val : mat) {
        val = dist(rng);
    }
}

// Compute GFLOPS given M, N, K dimensions and duration in milliseconds
inline double calculate_gflops(int M, int N, int K, double duration_ms) {
    if (duration_ms <= 0.0) return 0.0;
    // Dense GEMM performs 2 * M * N * K floating point operations (M*N*K multiplies + M*N*K adds)
    double total_flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    double seconds = duration_ms / 1000.0;
    return (total_flops / seconds) / 1e9; // Convert to GFLOPS (10^9 FLOPs/s)
}

// Verify output matrix C against reference C_ref with tolerance epsilon
inline bool verify_matrix(const std::vector<float>& C, const std::vector<float>& C_ref, int rows, int cols, double& max_error, float eps = 1e-3f) {
    if (C.size() != C_ref.size()) {
        max_error = -1.0;
        return false;
    }
    
    max_error = 0.0;
    bool match = true;
    for (size_t i = 0; i < C.size(); ++i) {
        double diff = std::abs(static_cast<double>(C[i]) - static_cast<double>(C_ref[i]));
        double ref_val = std::abs(static_cast<double>(C_ref[i]));
        double relative_diff = (ref_val > 1e-5) ? diff / ref_val : diff;
        
        if (diff > max_error) {
            max_error = diff;
        }
        if (relative_diff > eps && diff > eps) {
            match = false;
        }
    }
    return match;
}

// Function declarations for CPU matrix multiplication algorithms
void gemm_cpu_naive(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_cpu_cache_optimized(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_cpu_multithreaded(const float* A, const float* B, float* C, int M, int N, int K, int num_threads = 0);

} // namespace benchmark

#endif // MATRIX_UTILS_HPP
