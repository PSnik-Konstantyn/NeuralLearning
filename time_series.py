import math
import random

from math_functions import sigmoid, sigmoid_derivative


def normalize(data, min_val, max_val):
    return [(x - min_val) / (max_val - min_val) for x in data]


def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val


raw_data = [0.79, 3.84, 0.92, 4.50, 0.96, 5.51, 1.14, 5.32, 0.39, 4.99, 1.36, 5.81, 1.90]
test_inputs = [
    [4.99, 1.36, 5.81],
    [1.36, 5.81, 1.90],
    [5.81, 1.90, 4.79]
]
expected_outputs = [4.79, 1.41]

min_val = min(raw_data)
max_val = max(raw_data)

data = normalize(raw_data, min_val, max_val)
test_inputs = [normalize(inp, min_val, max_val) for inp in test_inputs]
expected_outputs = normalize(expected_outputs, min_val, max_val)

train_x = [data[i:i + 3] for i in range(len(data) - 3)]
train_y = [data[i + 3] for i in range(len(data) - 3)]

random.seed(42)

w_input_hidden = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
w_hidden_output = [random.uniform(-1, 1) for _ in range(3)]
b_hidden = [random.uniform(-1, 1) for _ in range(3)]
b_output = random.uniform(-1, 1)

learning_rate = 0.1
max_iterations = 10000

for epoch in range(max_iterations):
    total_error = 0

    for i in range(len(train_x)):
        inputs = train_x[i]
        expected = train_y[i]

        hidden_activations = [sigmoid(sum(inputs[k] * w_input_hidden[j][k] for k in range(3)) + b_hidden[j]) for j in
                              range(3)]
        output = sigmoid(sum(hidden_activations[j] * w_hidden_output[j] for j in range(3)) + b_output)

        error = expected - output
        total_error += error ** 2

        delta_output = error * sigmoid_derivative(output)
        delta_hidden = [delta_output * w_hidden_output[j] * sigmoid_derivative(hidden_activations[j]) for j in range(3)]

        for j in range(3):
            w_hidden_output[j] += learning_rate * delta_output * hidden_activations[j]
            b_hidden[j] += learning_rate * delta_hidden[j]
            for k in range(3):
                w_input_hidden[j][k] += learning_rate * delta_hidden[j] * inputs[k]
        b_output += learning_rate * delta_output

    if epoch % 100000 == 0:
        print(f"Epoch {epoch}, MSE: {total_error / len(train_x)}")

total_test_error = 0

for i, test_input in enumerate(test_inputs):
    hidden_activations = [sigmoid(sum(test_input[k] * w_input_hidden[j][k] for k in range(3)) + b_hidden[j]) for j in
                          range(3)]
    test_prediction = sigmoid(sum(hidden_activations[j] * w_hidden_output[j] for j in range(3)) + b_output)
    denormalized_prediction = denormalize(test_prediction, min_val, max_val)
    print(f"Predicted x{i + 13}: {denormalized_prediction}")
    if i < len(expected_outputs):
        total_test_error += (denormalized_prediction - denormalize(expected_outputs[i], min_val, max_val)) ** 2

print(f"Test MSE: {total_test_error / len(expected_outputs)}")
