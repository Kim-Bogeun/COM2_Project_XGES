#!/bin/bash

echo "XGES Benchmark Suite"
echo "===================="
echo ""

if [ ! -f "build/src-cpp/xges" ]; then
    echo "Error: xges not built. Run 'mkdir build && cd build && cmake .. && make' first"
    exit 1
fi

DATASETS=("samples/Data_var200_samples600.npy" "samples/Data_var800_samples2000.npy" "test_1000x2000.npy")

for dataset in "${DATASETS[@]}"; do
    if [ ! -f "$dataset" ]; then
        echo "Warning: $dataset not found, skipping..."
        continue
    fi
    
    echo "Testing: $dataset"
    echo "-------------------"
    
    echo "CPU:"
    ./build/src-cpp/xges --input "$dataset" 2>&1 | grep -E "(search completed|Score:)"
    
    echo "GPU Custom:"
    ./build/src-cpp/xges --batch --input "$dataset" 2>&1 | grep -E "(search completed|Score:)"
    
    echo "GPU cuBLAS:"
    ./build/src-cpp/xges --batch --backend cublas --input "$dataset" 2>&1 | grep -E "(search completed|Score:)"
    
    echo ""
done
