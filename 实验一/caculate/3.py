
import matplotlib.pyplot as plt

# Define the number of intervals and corresponding estimated π values
num_intervals = [1000, 10000, 100000, 1000000]
estimated_pi = [3.1416, 3.14159, 3.1416, 3.14159]  # Replace with your actual estimated π values

# Plot the relationship between the number of intervals and π's accuracy
plt.figure(figsize=(8, 6))
plt.plot(num_intervals, estimated_pi, marker='o', linestyle='-')
plt.xscale('log')  # Logarithmic scale for better visualization
plt.title('Estimated π vs. Number of Intervals')
plt.xlabel('Number of Intervals')
plt.ylabel('Estimated π')
plt.grid(True)
plt.show()