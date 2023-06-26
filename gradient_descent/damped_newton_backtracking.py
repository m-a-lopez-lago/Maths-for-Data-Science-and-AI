import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sympy

x = sympy.Symbol('x')

# Function to optimize
f_expr = sympy.atan(x)
f = sympy.lambdify(x, f_expr)

# Gradient of the function: ∇f(x)
gradient_expr = sympy.diff(f_expr, x)
gradient = sympy.lambdify(x, gradient_expr)

# Newton's method with damped backtracking
def damped_newton(x_initial, num_iterations, epsilon):
    x_values = [x_initial]
    divergent_x = [x_initial] #store the intermediate guesses for anim
    alpha_values = [1]

    for i in range(num_iterations):
        x_current = x_values[-1]
        alpha_current = 1

        f_current = f(x_current)
        gradient_current = gradient(x_current)

        x_new = x_current - alpha_current * f_current / gradient_current
        
        while np.linalg.norm(x_new) > np.linalg.norm(x_current):
            x_values.append(x_current)
            divergent_x.append(x_new)
            alpha_values.append(alpha_current)
            alpha_current /= 2
            x_new = x_current - alpha_current * f_current / gradient_current
        
        divergent_x.append(x_new)
        x_values.append(x_new)
        alpha_values.append(alpha_current)

        # Stopping criterion
        if alpha_current * np.linalg.norm(gradient_current) < \
        epsilon or np.abs(f(x_new) - f_current) < epsilon:
            break

    return x_values, alpha_values, divergent_x

# Parameters
x_initial = 10
num_iterations = 1000
epsilon = 1e-6

# Perform damped Newton's method
x_values, alpha_values, divergent_x_values = \
    damped_newton(x_initial, num_iterations, epsilon)

# Create figure and axis
fig, ax = plt.subplots()

# Set up the x-axis range
x_range = np.linspace(-25, 25, 400)

# Plot the function f(x)
ax.plot(x_range, np.zeros(len(x_range)), color='#ccc')

# Plot the function f(x)
ax.plot(x_range, f(x_range), color='gray')

# Initialize the point on the graph
point, = ax.plot(x_initial, f(x_initial), 'ro')

# Initialize the lines
divergent_line, = ax.plot([], [], 'b--', alpha=.3)
tangent_line, = ax.plot([], [], 'r')

# Initialize the dots for each guess with alpha = 0.5
dots, = ax.plot([], [], 'ro', alpha=0.5, markeredgecolor='none')

# Add legend
ax.plot([], [], color='gray', label='f(x) = arctan(x)')
ax.plot([], [], color='red', label='Updated tangent')
ax.plot([], [], color='blue', alpha=.3, linestyle='--', \
        label='backtracking iterations')
legend = ax.legend()

# Initialize the text annotation for alpha value
alpha_text = ax.text(0.05, 0.7, '', transform=ax.transAxes, ha='left')

# Initialize the dashed lines
dashed_lines = []

# Update function for animation
def update(i):
    point.set_data(x_values[i], f(x_values[i]))
    
    # Calculate tangent for backtracking iterations as it searches 
    tangent_slope_d = gradient(divergent_x_values[i])
    tangent_intercept_d = f(divergent_x_values[i]) - tangent_slope_d \
                                                   * divergent_x_values[i]
    tangent_y_d = tangent_slope_d * x_range + tangent_intercept_d

    divergent_line.set_data(x_range, tangent_y_d) 
    
    # Calculate tangent line coordinates
    tangent_slope = gradient(x_values[i])
    tangent_intercept = f(x_values[i]) - tangent_slope * x_values[i]
    tangent_y = tangent_slope * x_range + tangent_intercept

    tangent_line.set_data(x_range, tangent_y)   

    dots.set_data(x_values[:i+1], f(np.array(x_values[:i+1])))

    # Update alpha value text
    alpha_text.set_text(f'\u03B1 \u2248 {alpha_values[i]:.5f} \nx \u2248 {x_values[i]:.5f}')

    # Update dashed lines
    for line in dashed_lines:
        line.remove()
    dashed_lines.clear()
    for x_val in set(x_values[:i+1]):
        dashed_line = ax.plot([x_val, x_val], [0, f(x_val)], 'k--', alpha=0.5)
        dashed_lines.extend(dashed_line)

    return point, divergent_line, tangent_line, dots, alpha_text, dashed_lines

# Set up the animation
ani = FuncAnimation(fig, update, frames=len(x_values), interval=500, blit=False)

# Show the final plot
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title("Damped Newton's method with backtracking")
plt.show()

writer = PillowWriter(fps=5)

# Save the animation as an animated GIF
ani.save('damped_newton_optimization_animation.gif', writer=writer)
