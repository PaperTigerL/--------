import numpy as np
import matplotlib.pyplot as plt

# Define the execution times for different configurations
execution_times = {
    (2, 8): 1.327515205,
    (2, 16): 1.418978715,
    (2, 32): 1.377721686,
    (3, 8): 1.327515205,
    (3, 16): 1.418978715,
    (3, 32): 1.377721686,
    (4, 8): 1.327515205,
    (4, 16): 1.418978715,
    (4, 32): 1.377721686,
}

# Define the baseline execution time (e.g., for m=2, n=8)
baseline_time = execution_times[(2, 8)]

# Calculate speedup and efficiency
speedup = {}
efficiency = {}

for config, time in execution_times.items():
    m, n = config
    speedup[config] = baseline_time / time
    efficiency[config] = speedup[config] / (m * n)

# Create lists for x-axis values (block dimensions)
block_dims = [config[1] for config in execution_times.keys()]

# Create lists for y-axis values (speedup and efficiency)
speedup_values = [speedup[config] for config in execution_times.keys()]
efficiency_values = [efficiency[config] for config in execution_times.keys()]

# Plot speedup curve
plt.figure(figsize=(10, 5))
plt.plot(block_dims, speedup_values, marker='o', linestyle='-')
plt.title('Speedup vs Block Dimension')
plt.xlabel('Block Dimension (n)')
plt.ylabel('Speedup')
plt.grid(True)
plt.show()

# Plot efficiency curve
plt.figure(figsize=(10, 5))
plt.plot(block_dims, efficiency_values, marker='o', linestyle='-')
plt.title('Efficiency vs Block Dimension')
plt.xlabel('Block Dimension (n)')
plt.ylabel('Efficiency')
plt.grid(True)
plt.show()