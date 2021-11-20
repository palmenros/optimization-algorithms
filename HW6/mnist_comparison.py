import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Load dataframe using pandas
df = pd.read_csv("../HW3-5/mnist_train.csv", header=None)

samples_per_digit = 500
digit1 = 4
digit2 = 9

dataframe_0 = df[df.iloc[:, 0] == digit1].iloc[0:samples_per_digit, 1:].to_numpy()
dataframe_1 = df[df.iloc[:, 0] == digit2].iloc[0:samples_per_digit, 1:].to_numpy()

minus_ones = np.full(samples_per_digit, -1)
ones = np.full(samples_per_digit, 1)

# Array of x_i (image_data)
x = np.concatenate((dataframe_0, dataframe_1))

# Array of y_i (labels)
y = np.concatenate((minus_ones, ones))

# Total number of samples
N = x.shape[0]
d = x.shape[1]

# Initialize testing data
testing_df = pd.read_csv("../HW3-5/mnist_test.csv", header=None)
test_0 = testing_df[testing_df.iloc[:, 0] == digit1].iloc[0:samples_per_digit, 1:].to_numpy()
test_1 = testing_df[testing_df.iloc[:, 0] == digit2].iloc[0:samples_per_digit, 1:].to_numpy()

testing_data = np.concatenate((test_0, test_1))

total_testing_cases = testing_data.shape[0]
number_zero_testing_cases = test_0.shape[0]


def check_predictions_training_set(w):
    number_of_errors = 0
    for i in range(0, N):
        s = np.sign(w @ x[i])
        correct_sign = -1
        if i >= 500:
            correct_sign = 1

        if s != correct_sign:
            number_of_errors += 1

    return number_of_errors, N


def check_predictions_testing_set(w):
    number_of_testing_errors = 0

    for i in range(0, total_testing_cases):
        s = np.sign(w @ testing_data[i])
        correct_sign = -1
        if i >= number_zero_testing_cases:
            correct_sign = 1

        if s != correct_sign:
            number_of_testing_errors += 1

    return number_of_testing_errors, total_testing_cases


def gradient(w):
    exponential = np.exp((x @ w) * y)
    scalar = -y / (1 + exponential)
    return (x.T @ scalar) / N


def f(w):
    exponential = np.exp(-(x @ w) * y)
    return np.sum(np.log(1 + exponential)) / N


def regular_gradient_descent(max_iterations, initial_w, params=None):
    w = initial_w
    mu = 1e-6

    f_values = []

    for i in range(max_iterations):
        next_w = w - mu * gradient(w)

        # Store value of f(w) to create convergence rate plot
        f_values.append(f(w))

        w = next_w

    return w, f_values


def gradient_descent_momentum(max_iterations, initial_w, params):
    w = initial_w
    mu = params['mu']
    beta = params['beta']

    f_values = []
    momentum_direction = 0

    for i in range(max_iterations):
        new_momentum_direction = -mu * gradient(w) + beta * momentum_direction
        next_w = w + new_momentum_direction

        # Store value of f(w) to create convergence rate plot
        f_values.append(f(w))

        w = next_w
        momentum_direction = new_momentum_direction

    return w, f_values


def nesterov_acceleration(max_iterations, initial_w, params):
    w = initial_w
    mu = params['mu']
    beta = params['beta']

    f_values = []
    momentum_direction = 0

    for i in range(max_iterations):
        new_momentum_direction = beta * momentum_direction - mu * gradient(w + beta * momentum_direction)
        next_w = w + new_momentum_direction

        # Store value of f(w) to create convergence rate plot
        f_values.append(f(w))

        w = next_w
        momentum_direction = new_momentum_direction

    return w, f_values


def line_search_wolfe_conditions(w, p_t, c1, c2, beta, grad):
    a_t = 1
    number_iterations = 0
    while (f(w + a_t * p_t) > f(w) + c1 * a_t * (grad @ p_t) or np.abs(gradient(w + a_t * p_t) @ p_t) > -c2 * (grad @ p_t)) and number_iterations < 200:
        a_t = a_t * beta
        number_iterations += 1

    return a_t


def conjugate_gradient_fletcher_reeves(max_iterations, initial_w, params):
    w = initial_w

    f_values = []

    grad_t = gradient(w)
    p_t = -grad_t

    for i in range(max_iterations):
        alpha_t = line_search_wolfe_conditions(w, p_t, params['c1'], params['c2'], params['beta'], grad_t)
        next_w = w + alpha_t * p_t

        next_grad = gradient(next_w)
        beta = (next_grad @ next_grad) / (grad_t @ grad_t)
        p_t = -next_grad + beta * p_t

        # Store value of f(w) to create convergence rate plot
        f_values.append(f(w))

        w = next_w
        grad_t = next_grad

    return w, f_values


def conjugate_gradient_polak_ribiere(max_iterations, initial_w, params):
    w = initial_w

    f_values = []

    grad_t = gradient(w)
    p_t = -grad_t

    for i in range(max_iterations):
        alpha_t = line_search_wolfe_conditions(w, p_t, params['c1'], params['c2'], params['beta'], grad_t)
        next_w = w + alpha_t * p_t

        next_grad = gradient(next_w)
        beta = np.max((next_grad @ (next_grad - grad_t)) / (grad_t @ grad_t), 0)
        p_t = -next_grad + beta * p_t

        # Store value of f(w) to create convergence rate plot
        f_values.append(f(w))

        w = next_w
        grad_t = next_grad

    return w, f_values


def execute_method(max_iterations, descent_step_function, plot_function, params=None):
    w = np.zeros(d)

    print(f"Start value of f: {f(w)}")
    w, f_values = descent_step_function(max_iterations, w, params)
    print(f"End value of f: {f(w)}")

    # Plot resulting value of f(w) after each iteration

    plot_function(f_values, params)

    number_training_errors, number_training_tests = check_predictions_training_set(w)
    training_error_rate = number_training_errors / number_training_tests
    print(f'Error rate training set: {training_error_rate}, {number_training_errors} errors out of {number_training_tests}')

    number_testing_errors, number_testing_tests = check_predictions_testing_set(w)
    testing_error_rate = number_testing_errors / number_testing_tests
    print(f'Error rate testing set: {testing_error_rate}, {number_testing_errors} errors out of {number_testing_tests}')

    return {
        'training_error_rate': training_error_rate,
        'testing_error_rate': testing_error_rate,
        'final_w': f(w)
    }


def individual_plot(f_values, params=None):
    plt.plot(np.array(f_values))
    plt.xlabel('Number of iterations')
    plt.ylabel('$f(\omega)$')

    if params is not None and 'plot_title' in params.keys():
        plt.title(params['plot_title'])

    plt.show()


def together_logarithmic_plot(f_values, params=None):
    plt.plot(np.log(np.array(f_values)), label=params['title'])
    plt.xlabel('Number of iterations')
    plt.ylabel('$\log{f(\omega)}$')


def different_parameters_for_momentum_gradient_descent():
    result_list = []

    for mu in [1e-6, 5e-6, 1e-5, 5e-5]:
        for beta in [0.8, 0.85, 0.9, 0.95]:
            title = f"mu: {mu}, beta: {beta}"
            res = execute_method(300, gradient_descent_momentum, individual_plot, {'mu': mu, 'beta': beta, 'plot_title': title})
            res['title'] = title

            result_list.append(res)

    keys = ['training_error_rate', 'testing_error_rate', 'final_w']

    # Max
    for key in keys:
        res_max = max(result_list, key=lambda r: r[key])
        print(f'Max {key}: {res_max["title"]} with value {res_max[key]}')

    # Min
    for key in keys:
        res_min = min(result_list, key=lambda r: r[key])
        print(f'Min {key}: {res_min["title"]} with value {res_min[key]}')

    """
        Max training_error_rate: mu: 1e-06, beta: 0.8 with value 0.013
        Max testing_error_rate: mu: 1e-05, beta: 0.8 with value 0.064
        Max final_w: mu: 1e-06, beta: 0.8 with value 0.06411790069351866
        Min training_error_rate: mu: 1e-05, beta: 0.95 with value 0.0
        Min testing_error_rate: mu: 1e-06, beta: 0.95 with value 0.056
        Min final_w: mu: 5e-05, beta: 0.95 with value 0.00012238942176938844
    """


def different_parameters_for_nesterov():
    result_list = []

    for mu in [1e-6, 5e-6, 1e-5, 5e-5]:
        for beta in [0.8, 0.85, 0.9, 0.95]:
            title = f"mu: {mu}, beta: {beta}"
            res = execute_method(300, nesterov_acceleration, individual_plot, {'mu': mu, 'beta': beta, 'plot_title': title})
            res['title'] = title

            result_list.append(res)

    keys = ['training_error_rate', 'testing_error_rate', 'final_w']

    # Max
    for key in keys:
        res_max = max(result_list, key=lambda r: r[key])
        print(f'Max {key}: {res_max["title"]} with value {res_max[key]}')

    # Min
    for key in keys:
        res_min = min(result_list, key=lambda r: r[key])
        print(f'Min {key}: {res_min["title"]} with value {res_min[key]}')

    """
    Max training_error_rate: mu: 1e-06, beta: 0.8 with value 0.013
    Max testing_error_rate: mu: 1e-05, beta: 0.85 with value 0.064
    Max final_w: mu: 1e-06, beta: 0.8 with value 0.06429699915532149
    Min training_error_rate: mu: 1e-05, beta: 0.95 with value 0.0
    Min testing_error_rate: mu: 5e-05, beta: 0.8 with value 0.057
    Min final_w: mu: 5e-05, beta: 0.95 with value 0.00014404323671808792
    """


execute_method(300, regular_gradient_descent, together_logarithmic_plot, params={'title': 'Regular GD'})
execute_method(300, gradient_descent_momentum, together_logarithmic_plot, params={'mu': 5e-5, 'beta': 0.95, 'title': 'Momentum'})
execute_method(300, nesterov_acceleration, together_logarithmic_plot, params={'mu': 5e-5, 'beta': 0.95, 'title': 'Nesterov'})
execute_method(300, conjugate_gradient_fletcher_reeves, together_logarithmic_plot, params={'c1': 1/6, 'c2': 2/6, 'beta': 0.8, 'title': 'CG - FR'})
execute_method(300, conjugate_gradient_polak_ribiere, together_logarithmic_plot, params={'c1': 1/6, 'c2': 2/6, 'beta': 0.8, 'title': 'CG - PR'})

plt.legend()
plt.show()
