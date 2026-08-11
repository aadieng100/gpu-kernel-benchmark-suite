# ⚡ GPU Kernel Benchmark Suite

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus)](https://en.cppreference.com/w/cpp/20)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![WebGPU](https://img.shields.io/badge/WebGPU-WGSL-00f2fe?logo=webgpu)](https://www.w3.org/TR/webgpu/)
[![Build](https://img.shields.io/badge/CMake-3.20+-064F8C?logo=cmake)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A high-performance GPGPU benchmarking and kernel optimization suite evaluating **Dense Matrix Multiplication (GEMM)** across **C++20 CPU baselines**, **Modern CUDA GPU Kernels** (NVIDIA T4/A100), and **Cross-Platform WebGPU/WGSL Compute Shaders** running natively on **Apple Metal (macOS)**.

---

## 🎯 Executive Summary & Objectives

Designing ultra-low latency, high-throughput GPU kernels requires a deep understanding of memory hierarchies, thread warp scheduling, memory coalescing, on-chip SRAM caching, and instruction-level SIMD/vector parallelism.

This project demonstrates step-by-step kernel optimization techniques:
1. **CPU Baseline (C++20)**: Transforming cache-unfriendly stride-$N$ loops into $i$-$k$-$j$ cache-coherent SIMD loops and domain-decomposed multi-threading (**>100x speedup** over naive CPU).
2. **NVIDIA CUDA C++**: Exploiting Global Memory Coalescing, $32 \times 32$ **Shared Memory Tiling (`__shared__`)**, 128-bit `float4` vector loads (`LDG.E.128`), and $4 \times 4$ **2D Register Tiling**.
3. **WebGPU / WGSL**: Implementing Workgroup Shared Memory (`var<workgroup>`) and barrier synchronization (`workgroupBarrier()`) for zero-CUDA local execution on Apple Silicon (Metal API).
4. **Hardware Profiling**: Automated Nsight Compute (`ncu`) Speed-of-Light (SOL) analysis, Nsight Systems (`nsys`) timeline profiling, and WebGPU `timestamp-query` metrics.

---

## 📊 Performance Benchmarks & Results

### 1. CPU Baseline Benchmarks (Local Host - 8 Physical Cores)

| Matrix Size ($M \times N \times K$) | Implementation | Latency (ms) | Throughput (GFLOPS) | Speedup vs Naive | Numerical Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$1024 \times 1024 \times 1024$** | 1. CPU Naive ($i$-$j$-$k$) | 3873.32 ms | 0.55 GFLOPS | 1.00x | Reference |
| | 2. CPU Cache-Optimized ($i$-$k$-$j$) | 213.91 ms | 10.04 GFLOPS | **18.11x** | **PASSED** ($\text{max\_err} = 0.0$) |
| | 3. CPU Multi-Threaded (8 threads) | **32.70 ms** | **65.67 GFLOPS** | **118.45x** | **PASSED** ($\text{max\_err} = 0.0$) |

---

### 2. WebGPU WGSL Benchmarks (Local macOS - Apple Metal Backend)

| Matrix Size ($M \times N \times K$) | WGSL Shader Implementation | Latency (ms) | Throughput (GFLOPS) | Speedup vs Naive | Numerical Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$512 \times 512 \times 512$** | 1. WebGPU WGSL Naive (Direct Storage) | 114.80 ms | 2.34 GFLOPS | 1.00x | **PASSED** |
| | 2. WebGPU WGSL Workgroup Tiled ($16 \times 16$) | **28.50 ms** | **9.42 GFLOPS** | **4.03x** | **PASSED** |

---

### 3. CUDA GPU Benchmarks (Google Colab - NVIDIA T4 GPU)

| Matrix Size ($M \times N \times K$) | CUDA Kernel Implementation | Latency (ms) | Throughput (GFLOPS) | Speedup vs Naive | Numerical Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$2048 \times 2048 \times 2048$** | 1. Naive CUDA (Global Memory Direct) | 88.42 ms | 194.27 GFLOPS | 1.00x | **PASSED** |
| | 2. Shared Memory Tiled ($32 \times 32$) | 14.15 ms | 1213.78 GFLOPS | **6.25x** | **PASSED** |
| | 3. Vectorized `float4` + 2D Register Tiled | **8.12 ms** | **2114.53 GFLOPS** | **10.89x** | **PASSED** |

---

## 🧠 Architectural Deep-Dive & Hardware Concepts

```
+---------------------------------------------------------------------------------+
|                                 GPU MEMORY HIERARCHY                             |
+---------------------------------------------------------------------------------+
| Registers           | Private per Thread    | 0 Cycles Latency   | ~64K / SM    |
| Shared Memory (SRAM)| Shared per Thread Block| ~20 Cycles Latency  | ~48-100 KB   |
| L2 Cache            | Shared across all SMs | ~200 Cycles Latency | Several MB   |
| Global DRAM (VRAM)  | High-Capacity off-chip| ~400-800 Cycles     | 8 - 80 GB    |
+---------------------------------------------------------------------------------+
```

### Key Optimization Paradigms
- **Spatial Locality & Memory Coalescing**: Ensuring 32 threads in a Warp issue contiguous 128-byte Global Memory reads, saturating memory bus transactions.
- **Shared Memory Tiling**: Pre-caching $32 \times 32$ data tiles into on-chip SRAM, dropping DRAM accesses by $32\times$.
- **Bank Conflict Avoidance**: Structuring shared memory indexing so `threadIdx.x` steps across 32 distinct 4-byte memory banks without serialization.
- **2D Register Micro-Tiling**: Storing $4 \times 4 = 16$ intermediate accumulations per thread directly in zero-latency GPU registers.

---

## 📁 Repository Structure

```
gpu-kernel-benchmark-suite/
├── CMakeLists.txt                # C++20 & optional CUDA CMake configuration
├── .gitignore                    # Git rules (ignoring build/, private docs)
├── colab_runner.ipynb            # 1-Click Google Colab Jupyter Notebook
├── include/
│   ├── matrix_utils.hpp          # Matrix utilities, random init & GFLOPS timer
│   └── cuda_gemm.cuh             # CUDA header & C++ host wrapper signatures
├── src/
│   ├── cpu_baseline.cpp          # CPU GEMM (Naive, Cache-Optimized, Multi-Threaded)
│   ├── main_cpu.cpp              # CPU Benchmark Driver
│   ├── cuda_gemm.cu              # CUDA Kernels (Naive, Shared Tiled, Vectorized)
│   └── main_cuda.cu              # CUDA GPU Benchmark Driver
├── webgpu/
│   ├── index.html                # Interactive Dark-Mode WebGPU Dashboard
│   └── shaders/
│       ├── gemm_naive.wgsl       # Naive WGSL Compute Shader
│       └── gemm_tiled.wgsl       # Workgroup SRAM Tiled WGSL Compute Shader
└── scripts/
    ├── profile_runner.py         # Python automated benchmark & report parser
    └── run_ncu_profiling.sh      # Nsight Compute / Systems profiling runner
```

---

## 🛠️ Quickstart & Build Instructions

### 1. Build and Run CPU Baseline Locally (macOS / Linux / Windows)
```bash
# Compile using Clang++ / GCC C++20
mkdir -p build
clang++ -std=c++20 -O3 -march=native -Iinclude src/cpu_baseline.cpp src/main_cpu.cpp -o build/cpu_benchmark

# Execute CPU Benchmark
./build/cpu_benchmark
```

### 2. Run WebGPU Compute Shaders Locally on macOS (No CUDA Needed)
```bash
# Option A: Open directly in browser (Safari / Chrome / Edge)
open webgpu/index.html

# Option B: Serve via Python HTTP server
python3 -m http.server 8080 --directory webgpu
# Navigate to http://localhost:8080
```

### 3. Run CUDA GPU Kernels on Google Colab (NVIDIA T4 GPU)
1. Open [`colab_runner.ipynb`](colab_runner.ipynb) in Google Colab.
2. Select **Runtime -> Change runtime type -> GPU T4**.
3. Execute all cells to clone, compile via `nvcc`, and run CUDA benchmarks.

### 4. Run Automated Profiling Suite & Report Generation
```bash
./scripts/profile_runner.py
# Outputs: benchmark_results.json and benchmark_results.csv
```

---

## 📜 Technical Defense & System Documentation

The repository includes internal comprehensive guides (*git-ignored by design* for technical interview alignment):
- `EXPLICATION_FR.md`: In-depth French synthesis of SIMT execution, Warp scheduling, SRAM tiling, and Roofline Model metrics.
- `INTERVIEW_DEFENSE_EN.md`: English Q&A technical defense manual featuring 16+ senior GPU systems interview questions.

---

## 👤 Author & Acknowledgments

Developed by **Abdoul Aziz Dieng** as a portfolio project demonstrating expertise in CUDA C++, WebGPU WGSL, CPU cache optimization, and GPGPU systems performance engineering.
