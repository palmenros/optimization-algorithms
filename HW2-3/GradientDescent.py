import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def gradient_descent(u, v):
    # Get the length N of the data points
    N = len(u)

    # Construct the matrix A so we can calculate the matrix norm and compute f
    A = np.zeros((N, 2))
    A[:, 0] = u ** 2
    A[:, 1] = v ** 2

    # Construct vector b
    b = np.full(N, 1)

    # Function that evaluates f at point a
    def f(a):
        return (A.dot(a) - b).dot((A.dot(a) - b))

    # Calculate norm of A and mu
    norm = np.linalg.norm(A.transpose().dot(A), ord=2)
    mu = 1 / (2 * norm)

    # Start gradient descent
    a_t = np.array([1., 1.])
    should_stop = False

    # Minimum difference between iterations so that we continue the gradient descent algorithm
    epsilon = 1e-8
    current_iteration = 0

    print(f"Parameters: mu={mu}")
    print(f"Starting values: a=({a_t[0]}, {a_t[1]}), f(a) = {f(a_t)}")

    while not should_stop:
        # Iteration of gradient descent

        # The ith element of vec is a1*u_n^2 + a2*v_n^2 - 1
        vec = A.dot(a_t) - b

        # Clone vec as a column matrix with two columns to multiply element-wise with A
        B = np.array([vec, vec]).transpose()

        # Multiply A and B element wise and sum each column
        gradient = 2 * np.sum(A * B, axis=0)

        # Calculate next a_t
        next_a = a_t - mu * gradient

        # Print debug information about the algorithm in each iteration
        print(f"\tGradient: {gradient}")
        print(f"Iteration {current_iteration}, a=({a_t[0]}, {a_t[1]}), f(a) = {f(a_t)}")

        current_iteration += 1
        if abs(f(next_a) - f(a_t)) < epsilon:
            should_stop = True

        # End this iteration
        a_t = next_a
    return (current_iteration, a_t, f(a_t))

filename = "HW2-3/HW2-3.txt"

# Load dataframe using pandas
df = pd.read_csv(filename, header=None)

# Extract u and v coordinates from dataframe
u = df.iloc[0].to_numpy()
v = df.iloc[1].to_numpy()

(current_iteration, a_t, f_a_t) = gradient_descent(u, v)

print(f"Final values after {current_iteration} iterations: a=({a_t[0]}, {a_t[1]}), f(a) = {f_a_t}")

# Plot together the scatter plot with the ellipse obtained
plt.scatter(u, v)

# We are going to plot parametrically the ellipse, so we have to convert the radius
radius = 1 / np.sqrt(a_t)

# Plot the ellipse
t = np.linspace(0, 2 * np.pi, 100)
plt.plot(radius[0] * np.cos(t), radius[1] * np.sin(t), color='red')

# Plot a grid
plt.grid(color='lightgray', linestyle='--')

plt.show()

