#!/bin/bash

# 编译 CUDA 程序
nvcc test.cu -o test

# 固定 m 的值为 4
m=4

# 定义 n 和 block 的数组
n_values=(2 3 4)
block_values=(8 16 32)

# 遍历 n 和 block 的数组
for n in "${n_values[@]}"
do
    for block in "${block_values[@]}"
    do
        # 执行测试
        start_time=$(date +%s.%N)
        ./test $m $n $block $block
        end_time=$(date +%s.%N)

        # 计算执行时间
        execution_time=$(echo "$end_time - $start_time" | bc)

        # 输出执行时间和执行时的参数
        echo "Execution Time: $execution_time seconds"
        echo "Arguments: m=$m, n=$n, block_x=$block, block_y=$block"
        echo
    done
done