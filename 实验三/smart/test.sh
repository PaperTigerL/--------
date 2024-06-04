#!/bin/bash

# 编译并执行程序
nvcc -o test test.cpp -l OpenCL
./test

# 记录计算时间
start_time=$(date +%s%N)
./test
end_time=$(date +%s%N)
execution_time=$((end_time - start_time))
echo "程序执行时间：$((execution_time / 1000000)) 毫秒"