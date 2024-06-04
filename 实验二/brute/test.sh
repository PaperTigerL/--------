#!/bin/bash

# 编译代码
nvcc test.cu -o test

# 定义测试函数
function run_test {
    local m=$1
    local n=$2
    local block_size=$3

    echo "Running test with m=$m, n=$n, block_size=$block_size"

    # 执行程序并计算执行时间
    start_time=$(date +%s.%N)
    ./test $m $n $block_size
    end_time=$(date +%s.%N)
    execution_time=$(echo "$end_time - $start_time" | bc)

    echo "Execution time: $execution_time seconds"
    echo
}

# 定义参数组合
m=4
n_values=(2 4 6)
block_size_values=(16 32 64)

# 遍历参数组合
for n in "${n_values[@]}"; do
    for block_size in "${block_size_values[@]}"; do
        run_test $m $n $block_size
    done
done