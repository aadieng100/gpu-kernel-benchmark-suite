#include "matrix_utils.hpp"
#include <vector>
#include <thread>
#include <algorithm>

namespace benchmark {

/**
 * @brief Naive CPU Matrix Multiplication (i-j-k loop ordering).
 * 
 * Performance Bottleneck:
 * In the innermost loop over k, Matrix B is accessed as B[k * N + j].
 * Each step increments k, jumping across row memory boundaries by stride N.
 * This causes high L1/L2 cache misses since data brought into the cache line
 * (64 bytes = 16 floats) is mostly unused before being evicted.
 */
void gemm_cpu_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    // Zero out output matrix C
    std::fill(C, C + (M * N), 0.0f);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j]; // Stride N access on B!
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * @brief Cache-Optimized CPU Matrix Multiplication (i-k-j loop ordering).
 * 
 * Optimization Technique: Loop Interchange for Spatial Memory Locality.
 * In the innermost loop over j:
 * - A[i * K + k] is constant across iterations of j (cached in CPU register).
 * - B[k * N + j] is accessed sequentially in contiguous memory (stride 1).
 * - C[i * N + j] is updated sequentially in contiguous memory (stride 1).
 * 
 * Hardware Benefits:
 * 1. Exploits CPU L1/L2/L3 cache line prefetching (64-byte cache line = 16 floats).
 * 2. Allows vectorizers (AVX2, AVX-512, ARM NEON) to emit SIMD FMA instructions.
 */
void gemm_cpu_cache_optimized(const float* A, const float* B, float* C, int M, int N, int K) {
    // Zero out output matrix C
    std::fill(C, C + (M * N), 0.0f);

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a_ik = A[i * K + k]; // Broadcast scalar A[i,k]
            const float* b_row = &B[k * N];
            float* c_row = &C[i * N];
            
            #pragma omp simd // Hint compiler for SIMD auto-vectorization
            for (int j = 0; j < N; ++j) {
                c_row[j] += a_ik * b_row[j]; // Contiguous stride-1 access
            }
        }
    }
}

/**
 * @brief Multi-threaded Cache-Optimized CPU Matrix Multiplication.
 * 
 * Optimization Technique: Domain Decomposition across rows of M.
 * Spawns worker threads according to CPU core count. Each thread handles
 * a disjoint slice of rows [row_start, row_end) of output matrix C.
 * 
 * Hardware Benefits:
 * 1. Fully utilizes multi-core CPU architecture (e.g. 8-core / 10-core Apple Silicon / AMD EPYC / Intel Core).
 * 2. Eliminates false sharing and write conflicts: threads write to distinct memory regions.
 */
void gemm_cpu_multithreaded(const float* A, const float* B, float* C, int M, int N, int K, int num_threads) {
    if (num_threads <= 0) {
        num_threads = std::max(1u, std::thread::hardware_concurrency());
    }

    std::fill(C, C + (M * N), 0.0f);

    auto worker = [A, B, C, M, N, K](int row_start, int row_end) {
        for (int i = row_start; i < row_end; ++i) {
            for (int k = 0; k < K; ++k) {
                float a_ik = A[i * K + k];
                const float* b_row = &B[k * N];
                float* c_row = &C[i * N];
                
                for (int j = 0; j < N; ++j) {
                    c_row[j] += a_ik * b_row[j];
                }
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int rows_per_thread = M / num_threads;
    int remainder = M % num_threads;
    int current_row = 0;

    for (int t = 0; t < num_threads; ++t) {
        int start = current_row;
        int count = rows_per_thread + (t < remainder ? 1 : 0);
        int end = start + count;
        current_row = end;

        if (start < end) {
            threads.emplace_back(worker, start, end);
        }
    }

    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }
}

} // namespace benchmark
