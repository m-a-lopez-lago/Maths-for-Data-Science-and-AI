import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate some sample data
np.random.seed(42)
X = np.linspace(0, 10, 50)
y = 2 * 0.2 * X + 3 + np.random.randn(50) * 2

# Gradient descent optimization for linear regression
def gradient_descent(X, y, learning_rate, num_iterations):
    m, b = 0, 0  # Initial slope and intercept values
    n = len(X)  # Number of data points

    fig, ax = plt.subplots()
    scatter = ax.scatter(X, y, color='blue', label='Data Points')
    line, = ax.plot([], [], color='red', label='Fitted Line')

    # Set axis limits
    ax.set_xlim(0, 10)
    ax.set_ylim(np.min(y) - 1, np.max(y) + 1)

    def update(frame):
        nonlocal m, b

        # Compute predictions and errors
        y_pred = m * X + b
        error = y_pred - y + b

        # Update parameters
        m -= learning_rate * (2 / n) * np.dot(error, X)
        b -= learning_rate * (2 / n) * np.sum(error)

        # Update line data
        x_range = np.linspace(0, 10, 100)
        y_range = m * x_range + b
        line.set_data(x_range, y_range)

        return scatter, line

    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=num_iterations, interval=200, blit=True)

    # Add legend
    ax.legend()

    # Show the final plot
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression using Gradient Descent')

    # Display the animation
    plt.show()

# Parameters for optimization
learning_rate = 0.001
num_iterations = 1000

# Perform gradient descent
gradient_descent(X, y, learning_rate, num_iterations)
