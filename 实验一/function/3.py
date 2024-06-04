import matplotlib.pyplot as plt

# 区间数量
n_values = [256, 512, 1024]  # 替换为您的区间数量数据

# 执行时间
execution_time = [1.373239378, 1.356637564, 1.308607390]  # 替换为您的执行时间数据

# 绘制执行时间与区间数量关系图
plt.figure(figsize=(8, 6))
plt.plot(n_values, execution_time, marker='o', linestyle='-', color='r')
plt.xlabel('Number of Intervals')
plt.ylabel('Execution Time (seconds)')
plt.title('Relationship between Number of Intervals and Execution Time')
plt.grid(True)
plt.show()