#!/bin/bash

# Compile OpenCL program
 nvcc -o test test.cpp -l OpenCL

# Function to measure execution time
measure_time() {
    start_time=$SECONDS
    ./test
    duration=$((SECONDS - start_time))
    echo "Execution time: ${duration} seconds"
}

# Measure time for GPU execution
echo "=== GPU Execution ==="
measure_time

# Measure time for CPU execution
echo "=== CPU Execution ==="
export CL_DEVICE_TYPE=CPU
measure_time

# Clean up
rm test