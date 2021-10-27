import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Load dataframe using pandas
df = pd.read_csv("mnist_train.csv", header=None)

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

def gradient(w):
    exponential = np.exp((x @ w) * y)
    scalar = -y / (1 + exponential)
    return (x.T @ scalar) / N


def f(w):
    exponential = np.exp(-(x @ w) * y)
    return np.sum(np.log(1 + exponential)) / N


def gradient_descent_step(w, mu):
    #Given mu = 1e-6
    return w - mu * gradient(w)


def gradient_step_a(w, mu):
    #Given mu = 1e-8
    grad = gradient(w)
    p = -np.sign(grad) * np.sum(np.abs(grad))
    return w + mu * p


def gradient_step_b(w, mu):
    #Given mu = 1e-5
    grad = gradient(w)

    abs_grad = np.abs(grad)
    j = np.argmax(abs_grad)

    ej = np.zeros(d)
    ej[j] = 1

    p = -np.sign(grad[j]) * abs_grad[j] * ej
    return w + mu * p


def run_gradient_descent(step_function, mu, plot_color='b', plot_label=''):
    w = np.zeros(d)
    should_stop = False

    current_iteration = 0
    max_iterations = 500

    print(f"Start value of f: {f(w)}")

    f_values = []

    while not should_stop:
        current_iteration += 1
        next_w = step_function(w, mu)

        # Store value of f(w) to create convergence rate plot
        f_values.append(f(w))

        if current_iteration >= max_iterations:
            should_stop = True

        w = next_w

    print(f"End value of f: {f(w)}")

    # Plot resulting value of f(w) after each iteration

    plt.plot(np.array(f_values), color=plot_color, label=plot_label)
    plt.savefig('img/gradient_step_b_1e-5.png')


run_gradient_descent(gradient_descent_step, 1e-6, 'r', 'Standard GD ($\mu = 1e-6$)')
run_gradient_descent(gradient_step_a, 1e-8, 'b', '(a) ($\mu = 1e-8$)')
run_gradient_descent(gradient_step_b, 1e-4, 'g', '(b) ($\mu = 1e-4$)')

plt.xlabel('Number of iterations')
plt.legend()
plt.ylabel('$f(\omega)$')
plt.show()

exit()

# Try to predict from training set

number_of_errors = 0
for i in range(0, N):
    s = np.sign(w @ x[i])
    correct_sign = -1
    if i >= 500:
        correct_sign = 1

    if s != correct_sign:
        number_of_errors += 1

print(f'Error rate training set: {number_of_errors / N}, {number_of_errors} errors out of {N}')


# Try to predict from testing set

testing_df = pd.read_csv("mnist_test.csv", header=None)
test_0 = testing_df[testing_df.iloc[:, 0] == digit1].iloc[0:samples_per_digit, 1:].to_numpy()
test_1 = testing_df[testing_df.iloc[:, 0] == digit2].iloc[0:samples_per_digit, 1:].to_numpy()

testing_data = np.concatenate((test_0, test_1))

total_testing_cases = testing_data.shape[0]
number_zero_testing_cases = test_0.shape[0]

number_of_testing_errors = 0

for i in range(0, total_testing_cases):
    s = np.sign(w @ testing_data[i])
    correct_sign = -1
    if i >= number_zero_testing_cases:
        correct_sign = 1

    if s != correct_sign:
        number_of_testing_errors += 1


print(f'Error rate testing set: {number_of_testing_errors / total_testing_cases}, {number_of_testing_errors} errors out of {total_testing_cases}')
