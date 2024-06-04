import matplotlib.pyplot as plt

# Define the number of threads and corresponding execution times
num_threads = [32, 64, 128, 256]
execution_times = [1.398166080, 1.400590336, 1.369282845, 1.392424484]

# Calculate the efficiency relative to the execution time with 32 threads
efficiency = [execution_times[0] / (t * num) for t, num in zip(execution_times, num_threads)]

# Plot the efficiency curve
plt.figure(figsize=(8, 6))
plt.plot(num_threads, efficiency, marker='o', linestyle='-')
plt.title('Efficiency vs. Number of Threads')
plt.xlabel('Number of Threads')
plt.ylabel('Efficiency')
plt.grid(True)
plt.show()
