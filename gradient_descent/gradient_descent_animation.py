import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
from scipy.optimize import minimize

# Function to optimize: f(x) = x^4 - 4x^2 + x
def f(x):
    return x**4 - 4*x**2 + x

# Gradient of the function: ∇f(x) = 4x^3 - 8x + 1
def gradient(x):
    return 4*x**3 - 8*x + 1

# Basic gradient descent algo
def gradient_descent(x_initial, learning_rate, num_iterations):
    x_values = [x_initial]
    for i in range(num_iterations):
        x_new = x_values[-1] - learning_rate * gradient(x_values[-1])
        x_values.append(x_new)
    return x_values

# Parameters 
x_initial = 2
learning_rate = 0.01
num_iterations = 35

# Perform gradient descent
x_values = gradient_descent(x_initial, learning_rate, num_iterations)

# Create figure and axis
fig, ax = plt.subplots()

# Set up the x-axis range
x_range = np.linspace(-2, 2, 400)

# Plot the function f(x)
ax.plot(x_range, f(x_range), color='grey')

# Initialize the point on the graph
point, = ax.plot(x_initial, f(x_initial), 'ro')

# Initialize the line
line, = ax.plot([], [], 'r--')


# Add legend
ax.plot([], [], color='gray', label='f(x) = x$^4$ - 4$x^2$ + x')
ax.plot([], [], color='red', linestyle='--', label='Updated guesses')
ax.scatter([], [], color='blue', marker='o', label='Global minimum')
ax.legend(loc='upper left')

# get the real global min with scipy minimize
global_min = minimize(f,-2)

# Update function for animation
def update(i):
    point.set_data(x_values[i], f(x_values[i]))
    line.set_data(x_values[:i+1], f(np.array(x_values[:i+1])))
    ax.plot(global_min.x[0], global_min.fun, color='blue', marker='o', markersize=5)
    return point, line

# Set up the animation
ani = FuncAnimation(fig, update, frames=num_iterations, interval=500, blit=False)

# Show the final plot
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient descent missing the global optimum')
plt.show()

writer = PillowWriter(fps=8)

# Save the animation as an animated GIF
ani.save('gd_optimization_animation.gif', writer=writer)