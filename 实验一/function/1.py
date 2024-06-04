import matplotlib.pyplot as plt

# 数据
num_threads = [1, 2, 4, 8]  # 不同线程数
speedup = [1.0, 1.0 / 0.691, 1.0 / 0.598, 1.0 / 0.441]  # 速度提升数据
efficiency = [s / t for s, t in zip(speedup, num_threads)]  # 效率数据

# 绘制加速比曲线
plt.figure(figsize=(8, 6))
plt.plot(num_threads, speedup, marker='o', linestyle='-', color='b', label='Speedup')
plt.xlabel('Number of Threads')
plt.ylabel('Speedup')
plt.title('Speedup vs. Number of Threads')
plt.grid(True)
plt.legend()
plt.show()

# 绘制效率曲线
plt.figure(figsize=(8, 6))
plt.plot(num_threads, efficiency, marker='o', linestyle='-', color='g', label='Efficiency')
plt.xlabel('Number of Threads')
plt.ylabel('Efficiency')
plt.title('Efficiency vs. Number of Threads')
plt.grid(True)
plt.legend()
plt.show()

