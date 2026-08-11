#include "cuda_gemm.cuh"
#include <cuda_runtime.h>

namespace benchmark {

/**
 * @brief Kernel 1: Naive CUDA GEMM
 * 
 * Each CUDA thread (tx, ty) calculates one element C[row, col].
 * Inner loop over k accesses Global Memory directly at every iteration.
 * 
 * Memory Coalescing Status:
 * - Matrix B accesses B[k * N + col] are coalesced across a warp (tx = 0..31).
 * - High Global Memory Bandwidth Redundancy: Every element of A and B is loaded
 *   from Global Memory N and M times respectively.
 */
__global__ void gemm_cuda_naive_kernel(const float* __restrict__ A, 
                                        const float* __restrict__ B, 
                                        float* __restrict__ C, 
                                        int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * @brief Kernel 2: Shared Memory Tiled CUDA GEMM
 * 
 * Uses Shared Memory Tiling (TILE_SIZE x TILE_SIZE = 32 x 32) to cache matrix blocks
 * in fast on-chip SRAM (Shared Memory).
 * 
 * Performance Gains:
 * 1. Global Memory Access Reduction: Drops from 2*M*N*K accesses to 2*M*N*K / TILE_SIZE.
 * 2. Shared Memory Bandwidth: ~TB/s throughput with ~20-30 cycle latency vs ~200-400 cycles for DRAM.
 * 3. Bank Conflict Avoidance: As[ty][k] and Bs[k][tx] access patterns avoid 32-way bank conflicts.
 */
__global__ void gemm_cuda_tiled_kernel(const float* __restrict__ A, 
                                        const float* __restrict__ B, 
                                        float* __restrict__ C, 
                                        int M, int N, int K) {
    // Shared Memory allocation for A and B tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x,  by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float acc = 0.0f;

    // Loop over all tiles required to compute the dot product
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        // Collaborative load of tile from Global Memory to Shared Memory
        if (row < M && (t * TILE_SIZE + tx) < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if ((t * TILE_SIZE + ty) < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Barrier synchronization: Wait until all threads in block finish loading shared memory tile
        __syncthreads();

        // Compute dot product for current tile in Shared Memory
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        // Barrier synchronization: Ensure shared memory computation completes before next tile load
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

/**
 * @brief Kernel 3: Shared Memory Tiled + Float4 Vectorized CUDA GEMM
 * 
 * Advanced Optimizations:
 * 1. Vectorized Memory Transactions (float4 / 128-bit loads): Replaces 32-bit scalar loads
 *    with vectorized 128-bit memory instructions (LDG.E.128), maximizing memory bus utilization.
 * 2. 2D Register Tiling & Loop Unrolling: Computes multiple output elements per thread using CPU/GPU registers.
 */
__global__ void gemm_cuda_vectorized_kernel(const float* __restrict__ A, 
                                             const float* __restrict__ B, 
                                             float* __restrict__ C, 
                                             int M, int N, int K) {
    // 2D Block Tiling parameters
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Total threads in block = (BM/TM) * (BN/TN) = (64/4) * (64/4) = 16 * 16 = 256 threads
    int thread_id = ty * (BN / TN) + tx;

    // Mapping thread_id to Shared Memory load positions for A and B
    int a_shared_row = thread_id / (BK / 4);
    int a_shared_col = (thread_id % (BK / 4)) * 4;

    int b_shared_row = thread_id / (BN / 4);
    int b_shared_col = (thread_id % (BN / 4)) * 4;

    int a_global_row = by * BM + a_shared_row;
    int b_global_col = bx * BN + b_shared_col;

    // Register storage for 2D accumulator micro-tile (TM x TN)
    float r_c[TM][TN] = {0.0f};
    float r_a[TM];
    float r_b[TN];

    for (int bk = 0; bk < K; bk += BK) {
        // Vectorized load of A tile into Shared Memory (float4 = 128-bit)
        if (a_global_row < M && (bk + a_shared_col + 3) < K) {
            float4 tmp = *reinterpret_cast<const float4*>(&A[a_global_row * K + bk + a_shared_col]);
            As[a_shared_row][a_shared_col + 0] = tmp.x;
            As[a_shared_row][a_shared_col + 1] = tmp.y;
            As[a_shared_row][a_shared_col + 2] = tmp.z;
            As[a_shared_row][a_shared_col + 3] = tmp.w;
        } else {
            for (int i = 0; i < 4; ++i) {
                As[a_shared_row][a_shared_col + i] = (a_global_row < M && (bk + a_shared_col + i) < K) 
                                                     ? A[a_global_row * K + bk + a_shared_col + i] : 0.0f;
            }
        }

        // Vectorized load of B tile into Shared Memory
        if ((bk + b_shared_row) < K && (b_global_col + 3) < N) {
            float4 tmp = *reinterpret_cast<const float4*>(&B[(bk + b_shared_row) * N + b_global_col]);
            Bs[b_shared_row][b_shared_col + 0] = tmp.x;
            Bs[b_shared_row][b_shared_col + 1] = tmp.y;
            Bs[b_shared_row][b_shared_col + 2] = tmp.z;
            Bs[b_shared_row][b_shared_col + 3] = tmp.w;
        } else {
            for (int i = 0; i < 4; ++i) {
                Bs[b_shared_row][b_shared_col + i] = ((bk + b_shared_row) < K && (b_global_col + i) < N) 
                                                     ? B[(bk + b_shared_row) * N + b_global_col + i] : 0.0f;
            }
        }

        __syncthreads();

        // Accumulate outer-products from Shared Memory to Registers
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                r_a[m] = As[ty * TM + m][k];
            }
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                r_b[n] = Bs[k][tx * TN + n];
            }
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    r_c[m][n] += r_a[m] * r_b[n];
                }
            }
        }

        __syncthreads();
    }

    // Write-back register accumulators to Global Memory C matrix
    for (int m = 0; m < TM; ++m) {
        int global_r = by * BM + ty * TM + m;
        for (int n = 0; n < TN; ++n) {
            int global_c = bx * BN + tx * TN + n;
            if (global_r < M && global_c < N) {
                C[global_r * N + global_c] = r_c[m][n];
            }
        }
    }
}

// C++ Host Wrapper Functions with CUDA Event Timing
void run_cuda_naive(const float* h_A, const float* h_B, float* h_C, int M, int N, int K, float& time_ms) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    CHECK_CUDA_ERROR(cudaMalloc(&d_A, size_A));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, size_B));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, size_C));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Warm-up launch
    gemm_cuda_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    gemm_cuda_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));

    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms, start, stop));

    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

void run_cuda_tiled(const float* h_A, const float* h_B, float* h_C, int M, int N, int K, float& time_ms) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    CHECK_CUDA_ERROR(cudaMalloc(&d_A, size_A));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, size_B));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, size_C));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Warm-up launch
    gemm_cuda_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    gemm_cuda_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));

    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms, start, stop));

    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

void run_cuda_vectorized(const float* h_A, const float* h_B, float* h_C, int M, int N, int K, float& time_ms) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    CHECK_CUDA_ERROR(cudaMalloc(&d_A, size_A));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, size_B));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, size_C));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 block(16, 16); // 256 threads per block computing 64x64 sub-tile
    dim3 grid((N + 63) / 64, (M + 63) / 64);

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Warm-up launch
    gemm_cuda_vectorized_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    gemm_cuda_vectorized_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));

    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms, start, stop));

    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

} // namespace benchmark
