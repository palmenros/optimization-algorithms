import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Load dataframe using pandas
df = pd.read_csv("mnist_train.csv", header=None)

samples_per_digit = 500
digit1 = 0
digit2 = 1

dataframe_0 = df[df.iloc[:, 0] == digit1].iloc[0:samples_per_digit, 1:].to_numpy()
dataframe_1 = df[df.iloc[:, 0] == digit2].iloc[0:samples_per_digit, 1:].to_numpy()

minus_ones = np.full(samples_per_digit, -1)
ones = np.full(samples_per_digit, 1)

# Array of x_i (image_data)
x = np.concatenate((dataframe_0, dataframe_1)) / 255.0

# Array of y_i (labels)
y = np.concatenate((minus_ones, ones))

# Total number of samples
N = x.shape[0]
d = x.shape[1]

w = 2*np.random.rand(d) - 1
#w = np.zeros(d)
#w = np.full(d, 0.5)


def slow_gradient(w):
    res = np.zeros(d)
    for i in range(0, N):
        exponential = np.exp(-y[i] * (w @ x[i]))
        scalar = -y[i] * exponential / (1 + exponential)
        res += scalar * x[i]
    return res / N


#def fast_gradient(w):
#    exponential = np.exp(-(x @ w) * y)
#    scalar = -y * exponential / (1 + exponential)
#    return (x.T @ scalar) / N


def fast_gradient(w):
    exponential = np.exp((x @ w) * y)
    scalar = -y / (1 + exponential)
    return (x.T @ scalar) / N


def f(w):
    exponential = np.exp(-(x @ w) * y)
    return np.sum(np.log(1 + exponential)) / N


#mu = 1e-06
mu = 0.01
should_stop = False

current_iteration = 0
max_iterations = 300

print(f"Start value of f: {f(w)}")

f_values = []

while not should_stop:
    next_w = w - mu * fast_gradient(w)
    current_iteration += 1



    #print(f"Iteration {current_iteration}, f(w) = {f(w)}")
    f_values.append(f(w))

    if current_iteration >= max_iterations:
        should_stop = True

    w = next_w

print(f"End value of f: {f(w)}")

plt.plot(np.array(f_values))
plt.show()

# Try to predict from training set

number_of_success = 0
for i in range(0, N):
    s = np.sign(w @ x[i])
    correct_sign = -1
    if i >= 500:
        correct_sign = 1

    if s == correct_sign:
        number_of_success += 1

print(f'Success rate training set: {number_of_success / N}, {number_of_success} successes out of {N}')


# Try to predict from testing set
testing_df = pd.read_csv("mnist_test.csv", header=None)
test_0 = df[df.iloc[:, 0] == digit1].iloc[0:samples_per_digit, 1:].to_numpy()
test_1 = df[df.iloc[:, 0] == digit2].iloc[0:samples_per_digit, 1:].to_numpy()

testing_data = np.concatenate((test_0, test_1))

total_testing_cases = testing_data.shape[0]
number_zero_testing_cases = test_0.shape[0]

number_of_testing_success = 0

for i in range(0, total_testing_cases):
    s = np.sign(w @ x[i])
    correct_sign = -1
    if i >= number_zero_testing_cases:
        correct_sign = 1

    if s == correct_sign:
        number_of_testing_success += 1


print(f'Success rate testing set: {number_of_success / total_testing_cases}, {number_of_testing_success} successes out of {total_testing_cases}')
