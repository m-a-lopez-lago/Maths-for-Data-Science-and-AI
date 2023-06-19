import numpy as np
import matplotlib.pyplot as plt

# Define the function ||Ax - b||^2
def objective_function(x, A, b):
    return np.linalg.norm(np.dot(A, x) - b)**2

# Generate some random data for A and b
A = np.random.rand(50, 2)
b = np.random.rand(50)

# Create a grid of x values for plotting
x1 = np.linspace(-1, 1, 100)
x2 = np.linspace(-1, 1, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = np.zeros_like(X1)

# Compute the objective function for each point on the grid
for i in range(len(x1)):
    for j in range(len(x2)):
        x = np.array([x1[i], x2[j]])
        Z[i, j] = objective_function(x, A, b)

# Plot the 3D surface plot of the objective function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='jet')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Objective Function')
ax.set_title('Convexity of the Objective Function ||Ax - b||^2')
plt.show()
