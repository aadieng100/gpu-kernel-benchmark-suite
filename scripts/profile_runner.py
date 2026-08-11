#!/usr/bin/env python3
"""
Automated Benchmarking & Profiling Suite Parser for gpu-kernel-benchmark-suite
Generates JSON/CSV summary reports, Roofline Model metrics, and automated performance charts.
"""

import os
import sys
import json
import csv
import subprocess
import argparse
from datetime import datetime, timezone

def run_cpu_benchmark():
    print("==================================================================")
    print(" Running CPU Baseline Benchmark Suite...")
    print("==================================================================")
    cmd = ["./build/cpu_benchmark"]
    if not os.path.exists("./build/cpu_benchmark"):
        print("Error: ./build/cpu_benchmark executable not found. Build the project first.")
        return None

    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    return res.stdout

def run_ncu_profiling(exec_path="./build/cuda_benchmark"):
    print("==================================================================")
    print(" Generating NVIDIA Nsight Compute (ncu) Profiling Commands...")
    print("==================================================================")
    ncu_cmd = [
        "ncu",
        "--set", "full",
        "--metrics", "sm__throughput.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_active,smsp__shared_ld_bank_conflict.sum",
        "--target-processes", "all",
        exec_path
    ]
    print("Suggested Nsight Compute Command:")
    print(" ".join(ncu_cmd))
    return " ".join(ncu_cmd)

def generate_report_json(results_data, output_file="benchmark_results.json"):
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": {
            "platform": sys.platform,
            "python_version": sys.version
        },
        "benchmarks": results_data
    }
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Benchmark JSON report saved to: {output_file}")

def generate_csv_report(results_data, output_file="benchmark_results.csv"):
    headers = ["Backend", "Implementation", "MatrixSize", "Time_ms", "GFLOPS", "SpeedupVsNaive", "Verified"]
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in results_data:
            writer.writerow(row)
    print(f"[+] Benchmark CSV report saved to: {output_file}")

def parse_cpu_output(stdout):
    rows = []
    current_size = 512
    for line in stdout.splitlines():
        if "BENCHMARKING MATRIX SIZE:" in line:
            parts = line.split(":")
            if len(parts) > 1:
                size_str = parts[1].strip().split("x")[0].strip()
                current_size = int(size_str)
        elif "|" in line and "CPU" in line:
            cols = [c.strip() for c in line.split("|")[1:-1]]
            if len(cols) >= 5:
                impl_name = cols[0]
                time_ms = float(cols[1])
                gflops = float(cols[2])
                speedup = cols[3]
                verified = cols[4]
                rows.append(["CPU", impl_name, current_size, time_ms, gflops, speedup, verified])
    return rows

def main():
    parser = argparse.ArgumentParser(description="GPU Kernel Benchmark Profiling Runner")
    parser.add_argument("--run-cpu", action="store_true", help="Run CPU baseline benchmark and generate reports")
    parser.add_argument("--generate-ncu", action="store_true", help="Print Nsight Compute profiling commands")
    args = parser.parse_args()

    results = []
    if args.run_cpu or len(sys.argv) == 1:
        cpu_stdout = run_cpu_benchmark()
        if cpu_stdout:
            results.extend(parse_cpu_output(cpu_stdout))
            generate_report_json(results, "benchmark_results.json")
            generate_csv_report(results, "benchmark_results.csv")

    if args.generate_ncu:
        run_ncu_profiling()

if __name__ == "__main__":
    main()
