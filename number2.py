import numpy as np
import matplotlib.pyplot as plt

# Read matrix data from file
data = np.loadtxt("2 matrix.txt")
x = data[:, 0]
y = data[:, 1]

# Least Square Regression: y = mx + c
n = len(x)
m = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
c = (np.sum(y) - m * np.sum(x)) / n

print(f"m = {m}")
print(f"c = {c}")

# Generate regression line points
x_line = np.linspace(min(x), max(x), 100)
y_line = m * x_line + c

# Plot original data and regression line
plt.figure()
plt.scatter(x, y, color='red', label='Data points', marker='o')
plt.plot(x_line, y_line, color='blue', label='Least Square')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.tight_layout()
plt.show()
