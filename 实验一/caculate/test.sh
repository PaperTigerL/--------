#!/bin/bash

# Define the list of num_intervals and num_threads
num_intervals_list=(1000 10000 100000 1000000)
num_threads_list=(32 64 128 256)

# Compile the code
nvcc test.cu -o test

# Loop through num_intervals and num_threads
for num_intervals in "${num_intervals_list[@]}"
do
    for num_threads in "${num_threads_list[@]}"
    do
        # Run the code and measure execution time
        start_time=$(date +%s.%N)
        echo "Running with num_intervals=$num_intervals and num_threads=$num_threads"
        ./test $num_intervals $num_threads
        end_time=$(date +%s.%N)
        execution_time=$(echo "$end_time - $start_time" | bc)

        # Print the output and execution time
        echo "Output:"
        ./test $num_intervals $num_threads
        echo "Execution time: $execution_time seconds"
    done
done