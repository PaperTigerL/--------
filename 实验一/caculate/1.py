import matplotlib.pyplot as plt

# Define the number of threads and corresponding execution times
num_threads = [32, 64, 128, 256]
execution_times = [1.398166080, 1.400590336, 1.369282845, 1.392424484]

# Calculate the acceleration relative to the execution time with 32 threads
acceleration = [execution_times[0] / t for t in execution_times]

# Plot the acceleration curve
plt.figure(figsize=(8, 6))
plt.plot(num_threads, acceleration, marker='o', linestyle='-')
plt.title('Acceleration vs. Number of Threads')
plt.xlabel('Number of Threads')
plt.ylabel('Acceleration')
plt.grid(True)
plt.show()
