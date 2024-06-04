#!/bin/bash

# 编译CUDA代码
nvcc test.cu -o test

# 定义参数范围
n_values=(256 512 1024)
m_values=(1000000 2000000 3000000)

# 遍历参数组合
for n in "${n_values[@]}"
do
    for m in "${m_values[@]}"
    do
        # 执行测试并计算耗时
        start_time=$(date +%s.%N)
        ./test $n $m
        end_time=$(date +%s.%N)
        execution_time=$(echo "$end_time - $start_time" | bc)

        # 打印执行结果和耗时
        echo "n=$n, m=$m"
        echo "耗费时间：$execution_time 秒"
        echo "------------------------"
    done
done