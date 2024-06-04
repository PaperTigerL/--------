import matplotlib.pyplot as plt

# Define the number of intervals and corresponding execution times
num_intervals = [1000, 10000, 100000, 1000000]
execution_times = [1.398166080, 1.364095851, 1.379191564, 1.398029259]  # Replace with your actual execution times

# Plot the relationship between execution time and the number of intervals
plt.figure(figsize=(8, 6))
plt.plot(num_intervals, execution_times, marker='o', linestyle='-')
plt.xscale('log')  # Logarithmic scale for better visualization
plt.title('Execution Time vs. Number of Intervals')
plt.xlabel('Number of Intervals')
plt.ylabel('Execution Time (seconds)')
plt.grid(True)
plt.show()