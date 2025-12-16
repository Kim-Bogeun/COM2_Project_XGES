#!/bin/bash

# Benchmark script for XGES performance comparison
# Runs each configuration 5 times and calculates mean and std

EXECUTABLE="./build/src-cpp/xges"
SMALL_DATA="samples/Data_var20_avg_deg3.0_n_samples1000.npy"
SMALL_GRAPH="ground_truth/True_DAG_var20.csv"
MEDIUM_DATA="samples/Data_var400_samples1000.npy"
MEDIUM_GRAPH="ground_truth/True_DAG_var400.csv"
LARGE_DATA="samples/Data_var800_samples2000.npy"
LARGE_GRAPH="ground_truth/True_DAG_var800.csv"

RUNS=5
OUTPUT_FILE="benchmark_results.txt"
MD_FILE="benchmark_results.md"

# Function to extract execution time from output
extract_time() {
    grep "XGES search completed in" | sed -n 's/.*completed in \([0-9.]*\) seconds.*/\1/p'
}

# Function to calculate mean and std
calculate_stats() {
    python3 -c "
import sys
import numpy as np
times = [float(x) for x in sys.stdin.read().strip().split()]
print(f'Mean: {np.mean(times):.3f}s, Std: {np.std(times):.3f}s')
"
}

echo "========================================" | tee $OUTPUT_FILE
echo "XGES Performance Benchmark (5 runs each)" | tee -a $OUTPUT_FILE
echo "========================================" | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE

# Arrays to store all results for markdown generation
declare -A all_results

# Small Dataset (20 vars)
# CPU
echo "[CPU]" | tee -a $OUTPUT_FILE
cpu_times_small=""
for i in $(seq 1 $RUNS); do
    echo "  Run $i/5..." | tee -a $OUTPUT_FILE
    time=$($EXECUTABLE --input $SMALL_DATA --graph_truth $SMALL_GRAPH -v 1 2>&1 | extract_time)
    cpu_times_small="$cpu_times_small $time"
    echo "    Time: ${time}s" | tee -a $OUTPUT_FILE
done
result=$(echo $cpu_times_small | calculate_stats)
echo "  $result" | tee -a $OUTPUT_FILE
all_results["small_cpu"]="$result"
echo "" | tee -a $OUTPUT_FILE | tee -a $OUTPUT_FILE
done
echo "  $(echo $cpu_times_small | calculate_stats)" | tee -a $OUTPUT_FILE
# GPU Naive
echo "[GPU Naive]" | tee -a $OUTPUT_FILE
gpu_naive_times_small=""
for i in $(seq 1 $RUNS); do
    echo "  Run $i/5..." | tee -a $OUTPUT_FILE
    time=$($EXECUTABLE --input $SMALL_DATA --graph_truth $SMALL_GRAPH --gpu=true -v 1 2>&1 | extract_time)
    gpu_naive_times_small="$gpu_naive_times_small $time"
    echo "    Time: ${time}s" | tee -a $OUTPUT_FILE
done
result=$(echo $gpu_naive_times_small | calculate_stats)
echo "  $result" | tee -a $OUTPUT_FILE
all_results["small_naive"]="$result"
echo "" | tee -a $OUTPUT_FILE
echo "  $(echo $gpu_naive_times_small | calculate_stats)" | tee -a $OUTPUT_FILE
# GPU Optimized
echo "[GPU Optimized]" | tee -a $OUTPUT_FILE
gpu_opt_times_small=""
for i in $(seq 1 $RUNS); do
    echo "  Run $i/5..." | tee -a $OUTPUT_FILE
    time=$($EXECUTABLE --input $SMALL_DATA --graph_truth $SMALL_GRAPH --gpu=true --batch -v 1 2>&1 | extract_time)
    gpu_opt_times_small="$gpu_opt_times_small $time"
    echo "    Time: ${time}s" | tee -a $OUTPUT_FILE
done
result=$(echo $gpu_opt_times_small | calculate_stats)
echo "  $result" | tee -a $OUTPUT_FILE
all_results["small_opt"]="$result"
echo "" | tee -a $OUTPUT_FILE
echo "  $(echo $gpu_opt_times_small | calculate_stats)" | tee -a $OUTPUT_FILE
# GPU cuBLAS
echo "[GPU cuBLAS]" | tee -a $OUTPUT_FILE
gpu_cublas_times_small=""
for i in $(seq 1 $RUNS); do
    echo "  Run $i/5..." | tee -a $OUTPUT_FILE
    time=$($EXECUTABLE --input $SMALL_DATA --graph_truth $SMALL_GRAPH --gpu=true --batch --backend cublas -v 1 2>&1 | extract_time)
    gpu_cublas_times_small="$gpu_cublas_times_small $time"
    echo "    Time: ${time}s" | tee -a $OUTPUT_FILE
done
result=$(echo $gpu_cublas_times_small | calculate_stats)
echo "  $result" | tee -a $OUTPUT_FILE
all_results["small_cublas"]="$result"
echo "" | tee -a $OUTPUT_FILE

# Medium Dataset (400 vars)
echo "=== Medium Dataset (400 vars) ===" | tee -a $OUTPUT_FILE
# CPU
echo "[CPU]" | tee -a $OUTPUT_FILE
cpu_times_large=""
for i in $(seq 1 $RUNS); do
    echo "  Run $i/5..." | tee -a $OUTPUT_FILE
    time=$($EXECUTABLE --input $LARGE_DATA --graph_truth $LARGE_GRAPH -v 1 2>&1 | extract_time)
    cpu_times_large="$cpu_times_large $time"
    echo "    Time: ${time}s" | tee -a $OUTPUT_FILE
done
result=$(echo $cpu_times_large | calculate_stats)
echo "  $result" | tee -a $OUTPUT_FILE
all_results["large_cpu"]="$result"
echo "" | tee -a $OUTPUT_FILE
result=$(echo $cpu_times_medium | calculate_stats)
# GPU Naive - SKIPPED for large dataset
echo "[GPU Naive] SKIPPED (too slow for large dataset)" | tee -a $OUTPUT_FILE
all_results["large_naive"]="SKIPPED"
echo "" | tee -a $OUTPUT_FILE

# GPU Optimized
echo "[GPU Optimized]" | tee -a $OUTPUT_FILE
gpu_opt_times_large=""
for i in $(seq 1 $RUNS); do
    echo "  Run $i/5..." | tee -a $OUTPUT_FILE
    time=$($EXECUTABLE --input $LARGE_DATA --graph_truth $LARGE_GRAPH --gpu=true --batch -v 1 2>&1 | extract_time)
    gpu_opt_times_large="$gpu_opt_times_large $time"
    echo "    Time: ${time}s" | tee -a $OUTPUT_FILE
done
result=$(echo $gpu_opt_times_large | calculate_stats)
echo "  $result" | tee -a $OUTPUT_FILE
all_results["large_opt"]="$result"
echo "" | tee -a $OUTPUT_FILE

# GPU cuBLAS
echo "[GPU cuBLAS]" | tee -a $OUTPUT_FILE
gpu_cublas_times_large=""
for i in $(seq 1 $RUNS); do
    echo "  Run $i/5..." | tee -a $OUTPUT_FILE
    time=$($EXECUTABLE --input $LARGE_DATA --graph_truth $LARGE_GRAPH --gpu=true --batch --backend cublas -v 1 2>&1 | extract_time)
    gpu_cublas_times_large="$gpu_cublas_times_large $time"
    echo "    Time: ${time}s" | tee -a $OUTPUT_FILE
done
result=$(echo $gpu_cublas_times_large | calculate_stats)
echo "  $result" | tee -a $OUTPUT_FILE
all_results["large_cublas"]="$result"
echo "" | tee -a $OUTPUT_FILE

echo "========================================" | tee -a $OUTPUT_FILE
echo "Benchmark completed! Results saved to $OUTPUT_FILE" | tee -a $OUTPUT_FILE
echo "========================================" | tee -a $OUTPUT_FILE

# Generate Markdown file
echo "# XGES Performance Benchmark Results" > $MD_FILE
echo "" >> $MD_FILE
echo "Benchmark Date: $(date)" >> $MD_FILE
echo "" >> $MD_FILE
echo "Each configuration was run 5 times. Results show mean ± standard deviation." >> $MD_FILE
echo "" >> $MD_FILE

echo "## Dataset Information" >> $MD_FILE
echo "" >> $MD_FILE
echo "- **Small**: 20 variables × 1000 samples (\`$SMALL_DATA\`)" >> $MD_FILE
echo "- **Medium**: 400 variables × 1000 samples (\`$MEDIUM_DATA\`)" >> $MD_FILE
echo "- **Large**: 800 variables × 2000 samples (\`$LARGE_DATA\`)" >> $MD_FILE
echo "" >> $MD_FILE

echo "## Results Summary" >> $MD_FILE
echo "" >> $MD_FILE
echo "| Dataset | CPU | GPU Naive | GPU Optimized | GPU cuBLAS |" >> $MD_FILE
echo "|---------|-----|-----------|---------------|------------|" >> $MD_FILE
echo "| Small (20 vars) | ${all_results[small_cpu]} | ${all_results[small_naive]} | ${all_results[small_opt]} | ${all_results[small_cublas]} |" >> $MD_FILE
echo "| Medium (400 vars) | ${all_results[medium_cpu]} | ${all_results[medium_naive]} | ${all_results[medium_opt]} | ${all_results[medium_cublas]} |" >> $MD_FILE
echo "| Large (800 vars) | ${all_results[large_cpu]} | ${all_results[large_naive]} | ${all_results[large_opt]} | ${all_results[large_cublas]} |" >> $MD_FILE
echo "" >> $MD_FILE

echo "## Detailed Results" >> $MD_FILE
echo "" >> $MD_FILE
echo "### Small Dataset (20 variables)" >> $MD_FILE
echo "" >> $MD_FILE
echo "- **CPU**: ${all_results[small_cpu]}" >> $MD_FILE
echo "- **GPU Naive**: ${all_results[small_naive]}" >> $MD_FILE
echo "- **GPU Optimized**: ${all_results[small_opt]}" >> $MD_FILE
echo "- **GPU cuBLAS**: ${all_results[small_cublas]}" >> $MD_FILE
echo "" >> $MD_FILE

echo "### Medium Dataset (400 variables)" >> $MD_FILE
echo "" >> $MD_FILE
echo "- **CPU**: ${all_results[medium_cpu]}" >> $MD_FILE
echo "- **GPU Naive**: ${all_results[medium_naive]}" >> $MD_FILE
echo "- **GPU Optimized**: ${all_results[medium_opt]}" >> $MD_FILE
echo "- **GPU cuBLAS**: ${all_results[medium_cublas]}" >> $MD_FILE
echo "" >> $MD_FILE

echo "### Large Dataset (800 variables)" >> $MD_FILE
echo "" >> $MD_FILE
echo "- **CPU**: ${all_results[large_cpu]}" >> $MD_FILE
echo "- **GPU Naive**: ${all_results[large_naive]}" >> $MD_FILE
echo "- **GPU Optimized**: ${all_results[large_opt]}" >> $MD_FILE
echo "- **GPU cuBLAS**: ${all_results[large_cublas]}" >> $MD_FILE
echo "" >> $MD_FILE

echo "## Notes" >> $MD_FILE
echo "" >> $MD_FILE
echo "- GPU Naive implementation was skipped for the large dataset due to excessive runtime (>300s expected)" >> $MD_FILE
echo "- All experiments were conducted on the same hardware configuration" >> $MD_FILE
echo "- Verbosity level was set to 1 to capture timing information" >> $MD_FILE
echo "" >> $MD_FILE

echo "Results also saved to: $MD_FILE"

# GPU cuBLAS
echo "[GPU cuBLAS]" | tee -a $OUTPUT_FILE
gpu_cublas_times_medium=""
for i in $(seq 1 $RUNS); do
    echo "  Run $i/5..." | tee -a $OUTPUT_FILE
    time=$($EXECUTABLE --input $MEDIUM_DATA --graph_truth $MEDIUM_GRAPH --gpu=true --batch --backend cublas -v 1 2>&1 | extract_time)
    gpu_cublas_times_medium="$gpu_cublas_times_medium $time"
    echo "    Time: ${time}s" | tee -a $OUTPUT_FILE
done
result=$(echo $gpu_cublas_times_medium | calculate_stats)
echo "  $result" | tee -a $OUTPUT_FILE
all_results["medium_cublas"]="$result"
echo "" | tee -a $OUTPUT_FILE

# Large Dataset (800 vars)
echo "=== Large Dataset (800 vars) ===" | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE
echo "=== Large Dataset (1000 vars) ===" | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE

# CPU
echo "[CPU]" | tee -a $OUTPUT_FILE
cpu_times_large=""
for i in $(seq 1 $RUNS); do
    echo "  Run $i/5..." | tee -a $OUTPUT_FILE
    time=$($EXECUTABLE --input $LARGE_DATA --graph_truth $LARGE_GRAPH -v 0 2>&1 | extract_time)
    cpu_times_large="$cpu_times_large $time"
    echo "    Time: ${time}s" | tee -a $OUTPUT_FILE
done
echo "  $(echo $cpu_times_large | calculate_stats)" | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE

# GPU Naive - SKIPPED for large dataset
echo "[GPU Naive] SKIPPED (too slow for large dataset)" | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE

# GPU Optimized
echo "[GPU Optimized]" | tee -a $OUTPUT_FILE
gpu_opt_times_large=""
for i in $(seq 1 $RUNS); do
    echo "  Run $i/5..." | tee -a $OUTPUT_FILE
    time=$($EXECUTABLE --input $LARGE_DATA --graph_truth $LARGE_GRAPH --gpu=true --batch -v 0 2>&1 | extract_time)
    gpu_opt_times_large="$gpu_opt_times_large $time"
    echo "    Time: ${time}s" | tee -a $OUTPUT_FILE
done
echo "  $(echo $gpu_opt_times_large | calculate_stats)" | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE

# GPU cuBLAS
echo "[GPU cuBLAS]" | tee -a $OUTPUT_FILE
gpu_cublas_times_large=""
for i in $(seq 1 $RUNS); do
    echo "  Run $i/5..." | tee -a $OUTPUT_FILE
    time=$($EXECUTABLE --input $LARGE_DATA --graph_truth $LARGE_GRAPH --gpu=true --batch --backend cublas -v 0 2>&1 | extract_time)
    gpu_cublas_times_large="$gpu_cublas_times_large $time"
    echo "    Time: ${time}s" | tee -a $OUTPUT_FILE
done
echo "  $(echo $gpu_cublas_times_large | calculate_stats)" | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE

echo "========================================" | tee -a $OUTPUT_FILE
echo "Benchmark completed! Results saved to $OUTPUT_FILE" | tee -a $OUTPUT_FILE
echo "========================================" | tee -a $OUTPUT_FILE
