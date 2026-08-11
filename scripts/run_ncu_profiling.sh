#!/usr/bin/env bash
# Script to run NVIDIA Nsight Compute (ncu) & Nsight Systems (nsys) profiling on CUDA kernels

set -e

echo "========================================================================"
echo " NVIDIA NSIGHT COMPUTE & SYSTEMS AUTOMATED PROFILING SCRIPT"
echo "========================================================================"

BUILD_DIR="build"
CUDA_BIN="${BUILD_DIR}/cuda_benchmark"

if [ ! -f "${CUDA_BIN}" ]; then
    echo "CUDA binary not found. Compiling with NVCC..."
    mkdir -p "${BUILD_DIR}"
    nvcc -O3 -std=c++17 --use_fast_math -arch=sm_75 -Iinclude src/cpu_baseline.cpp src/cuda_gemm.cu src/main_cuda.cu -o "${CUDA_BIN}"
fi

echo ""
echo "1. Running Nsight Systems (nsys) Timeline Profile..."
nsys profile --stats=true --force-overwrite=true -o profile_timeline "${CUDA_BIN}" || echo "nsys not found or insufficient privileges."

echo ""
echo "2. Running Nsight Compute (ncu) Kernel Speed-of-Light (SOL) Analysis..."
ncu --set full \
    --metrics sm__throughput.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_active,smsp__shared_ld_bank_conflict.sum \
    --export profile_ncu_report \
    --force-overwrite \
    "${CUDA_BIN}" || echo "ncu not found or insufficient privileges."

echo ""
echo "Profiling complete! Inspection reports saved as profile_timeline.nsys-rep and profile_ncu_report.ncu-rep"
