import random

from math_functions import sigmoid_derivative, sigmoid

train_x = [(0, 0), (0, 1), (1, 0), (1, 1)]
train_y = [0, 1, 1, 0]

random.seed(42)
w_input_hidden = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
w_hidden_output = [random.uniform(-1, 1) for _ in range(2)]
b_hidden = [random.uniform(-1, 1) for _ in range(2)]
b_output = random.uniform(-1, 1)

learning_rate = 0.5
max_iterations = 10000
tolerance = 0.001

for epoch in range(max_iterations):
    total_error = 0

    for i in range(len(train_x)):
        inputs = train_x[i]
        expected_output = train_y[i]

        hidden_activations = [sigmoid(sum(inputs[k] * w_input_hidden[j][k] for k in range(2)) + b_hidden[j]) for j in
                              range(2)]
        output = sigmoid(sum(hidden_activations[j] * w_hidden_output[j] for j in range(2)) + b_output)

        error = expected_output - output
        total_error += error ** 2

        delta_output = error * sigmoid_derivative(output)
        delta_hidden = [delta_output * w_hidden_output[j] * sigmoid_derivative(hidden_activations[j]) for j in range(2)]

        for j in range(2):
            w_hidden_output[j] += learning_rate * delta_output * hidden_activations[j]
            b_hidden[j] += learning_rate * delta_hidden[j]
            for k in range(2):
                w_input_hidden[j][k] += learning_rate * delta_hidden[j] * inputs[k]
        b_output += learning_rate * delta_output

    if total_error < tolerance:
        break


def test_network(function_name, test_data, expected_results):
    print(f"Testing {function_name} function:")
    for i, inputs in enumerate(test_data):
        hidden_activations = [sigmoid(sum(inputs[k] * w_input_hidden[j][k] for k in range(len(inputs))) + b_hidden[j])
                              for j in range(2)]
        output = round(sigmoid(sum(hidden_activations[j] * w_hidden_output[j] for j in range(2)) + b_output))
        print(f"Input: {inputs} -> Output: {output} (Expected: {expected_results[i]})")
    print()


test_network("XOR", train_x, train_y)

