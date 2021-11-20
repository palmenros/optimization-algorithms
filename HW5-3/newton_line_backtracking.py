import numpy as np
import matplotlib.pyplot as plt


def f(x):
    x1, x2 = x[0], x[1]
    return 200 * (x2 - x1**2)**2 + (1-x1)**2


def grad(x):
    x1, x2 = x[0], x[1]
    return np.array([-800 * x1 * (x2 - x1**2) - 2 * (1-x1), 400 * (x2 - x1**2)])


def hessian(x):
    x1, x2 = x[0], x[1]
    return np.array([[2400 * x1**2 - 800*x2 + 2, -800 * x1], [-800 * x1, 400]])


def newton(x0, num_iterations):
    x = x0
    norm_x = []
    f_vals = []

    iter_number = 0
    while iter_number < num_iterations:

        p = np.linalg.solve(hessian(x), -grad(x))
        next_x = x + p

        #print(f'Iteration {iter_number}: x={x}, f(x)={f(x)}')
        norm_x.append(np.linalg.norm(x - np.array([1, 1]), ord=2))
        f_vals.append(f(x))

        iter_number += 1
        x = next_x

    norm_x.append(np.linalg.norm(x - np.array([1, 1]), ord=2))
    f_vals.append(f(x))
    #print(f'Final values: x={x}, f(x)={f(x)}')
    return norm_x, f_vals


def gradient_descent(x0, num_iterations):
    x = x0
    mu = 1e-3

    norm_x = []
    f_vals = []

    iter_number = 0
    while iter_number < num_iterations:
        next_x = x - mu * grad(x)
        #print(f'Iteration {iter_number}: x={x}, f(x)={f(x)}')
        norm_x.append(np.linalg.norm(x - np.array([1, 1]), ord=2))
        f_vals.append(f(x))

        iter_number += 1
        x = next_x

    norm_x.append(np.linalg.norm(x - np.array([1, 1]), ord=2))
    f_vals.append(f(x))
    #print(f'Final values: x={x}, f(x)={f(x)}')
    return norm_x, f_vals


def backtracking_line_search(x0, num_iterations):
    x = x0
    gamma = 1/4
    beta = 1/2

    norm_x = []
    f_vals = []

    iter_number = 0
    while iter_number < num_iterations:
        gradient = grad(x)
        mu = 1
        next_x = x - mu * grad(x)

        # Check if Armijo condition works
        while f(next_x) > f(x) - mu * gamma * gradient.dot(gradient):
            mu = mu * beta
            next_x = x - mu * grad(x)

        norm_x.append(np.linalg.norm(x - np.array([1, 1]), ord=2))
        f_vals.append(f(x))
        #print(f'Iteration {iter_number}: x={x}, f(x)={f(x)}, mu={mu}')

        iter_number += 1
        x = next_x

    norm_x.append(np.linalg.norm(x - np.array([1, 1]), ord=2))
    f_vals.append(f(x))
    #print(f'Final values: x={x}, f(x)={f(x)}')
    return norm_x, f_vals


number_iterations = 1000

x0 = np.random.random(2)
print(x0)

norm_x_newton, f_vals_newton = newton(x0, number_iterations)
norm_x_gd, f_vals_gd = gradient_descent(x0, number_iterations)
norm_x_bt, f_vals_bt = backtracking_line_search(x0, number_iterations)

plt.plot(np.array(norm_x_newton), color='r', label='Newton')
plt.plot(np.array(norm_x_gd), color='g', label='Gradient Descent, $\mu=10^{-3}$')
plt.plot(np.array(norm_x_bt), color='b', label='Backtracking Line Search')
plt.xlabel('Number of iterations')
plt.ylabel('$||x^{(t)}-x^{*}||$')
plt.legend()
plt.show()

plt.plot(np.array(f_vals_newton), color='r', label='Newton')
plt.plot(np.array(f_vals_gd), color='g', label='Gradient Descent, $\mu=10^{-3}$')
plt.plot(np.array(f_vals_bt), color='b', label='Backtracking Line Search')
plt.xlabel('Number of iterations')
plt.ylabel('$f(x^{(t)})-f(x^{*})$')
plt.legend()
plt.show()

plt.plot(np.array(f_vals_newton)[10:], color='r', label='Newton')
plt.plot(np.array(f_vals_gd)[10:], color='g', label='Gradient Descent, $\mu=10^{-3}$')
plt.plot(np.array(f_vals_bt)[10:], color='b', label='Backtracking Line Search')
plt.xlabel('Number of iterations')
plt.ylabel('$f(x^{(t)})-f(x^{*})$')
plt.legend()
plt.show()

