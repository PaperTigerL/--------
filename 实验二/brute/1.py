import matplotlib.pyplot as plt

# Define the execution times for different configurations
configurations = [
    {"m": 4, "n": 2, "block_size": 16, "execution_time": 1.393227087},
    {"m": 4, "n": 2, "block_size": 32, "execution_time": 1.376817058},
    {"m": 4, "n": 2, "block_size": 64, "execution_time": 1.370207666},
    {"m": 4, "n": 4, "block_size": 16, "execution_time": 1.367646798},
    {"m": 4, "n": 4, "block_size": 32, "execution_time": 1.385445766},
    {"m": 4, "n": 4, "block_size": 64, "execution_time": 1.388739555},
    {"m": 4, "n": 6, "block_size": 16, "execution_time": 1.378965127},
    {"m": 4, "n": 6, "block_size": 32, "execution_time": 1.374766293},
    {"m": 4, "n": 6, "block_size": 64, "execution_time": 1.355089437}
]

# Calculate the speedup ratios and efficiency ratios
baseline_time = configurations[0]["execution_time"]
speedup_ratios = [baseline_time / config["execution_time"] for config in configurations]
efficiency_ratios = [speedup / config["m"] for speedup, config in zip(speedup_ratios, configurations)]

# Extract block sizes for labels
block_sizes = [config["block_size"] for config in configurations]

# Plot speedup ratios
plt.figure(figsize=(10, 5))
plt.plot(block_sizes, speedup_ratios, marker='o', linestyle='-')
plt.xlabel('Block Size')
plt.ylabel('Speedup Ratio')
plt.title('Speedup Ratio vs Block Size')
plt.grid(True)
plt.xticks(block_sizes)
plt.tight_layout()
plt.savefig('speedup_ratio.png')
plt.show()

# Plot efficiency ratios
plt.figure(figsize=(10, 5))
plt.plot(block_sizes, efficiency_ratios, marker='o', linestyle='-')
plt.xlabel('Block Size')
plt.ylabel('Efficiency Ratio')
plt.title('Efficiency Ratio vs Block Size')
plt.grid(True)
plt.xticks(block_sizes)
plt.tight_layout()
plt.savefig('efficiency_ratio.png')
plt.show()