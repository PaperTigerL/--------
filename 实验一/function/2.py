import matplotlib.pyplot as plt

# 区间数量
n_values = [256, 512, 1024]  # 替换为您的区间数量数据

# 对应的π估计值
pi_values = [3.1415916536, 3.1415921536, 3.1415923203]  # 替换为您的π估计值数据

# 绘制区间数量与π精确度关系图
plt.figure(figsize=(8, 6))
plt.plot(n_values, pi_values, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Intervals')
plt.ylabel('Estimated Value of Pi')
plt.title('Relationship between Number of Intervals and Pi Accuracy')
plt.grid(True)
plt.show()