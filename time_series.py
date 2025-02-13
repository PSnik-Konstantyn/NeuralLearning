import random
from math_functions import sigmoid, sigmoid_derivative

def normalize(data, min_val, max_val):
    return [(x - min_val) / (max_val - min_val) for x in data]

def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val

raw_data = [1.59, 5.73, 0.48, 5.28, 1.35, 5.91, 0.77, 5.25, 1.37, 4.42, 0.26, 4.21, 1.90]
test_input = [4.21, 1.90, 4.08]
expected_output = 1.40

min_val, max_val = min(raw_data + test_input + [expected_output]), max(raw_data + test_input + [expected_output])
data = normalize(raw_data, min_val, max_val)
test_input = normalize(test_input, min_val, max_val)
expected_output = normalize([expected_output], min_val, max_val)[0]

train_x = [data[i:i + 3] for i in range(len(data) - 3)]
train_y = [data[i + 3] for i in range(len(data) - 3)]

random.seed(42)

w_input_hidden = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
w_hidden_output = [random.uniform(-1, 1) for _ in range(3)]
b_hidden = [random.uniform(-1, 1) for _ in range(3)]
b_output = random.uniform(-1, 1)

learning_rate = 0.01
max_iterations = 10000000
tolerance = 0.01

for epoch in range(max_iterations):
    total_error = 0
    w_input_hidden_updates = [[0] * 3 for _ in range(3)]
    w_hidden_output_updates = [0] * 3
    b_hidden_updates = [0] * 3
    b_output_update = 0

    for i in range(len(train_x)):
        inputs = train_x[i]
        expected = train_y[i]

        hidden_activations = [sigmoid(sum(inputs[k] * w_input_hidden[j][k] for k in range(3)) + b_hidden[j]) for j in range(3)]
        output = sigmoid(sum(hidden_activations[j] * w_hidden_output[j] for j in range(3)) + b_output)

        error = expected - output
        total_error += error ** 2

        delta_output = error * sigmoid_derivative(output)
        delta_hidden = [delta_output * w_hidden_output[j] * sigmoid_derivative(hidden_activations[j]) for j in range(3)]

        for j in range(3):
            w_hidden_output_updates[j] += delta_output * hidden_activations[j]
            b_hidden_updates[j] += delta_hidden[j]
            for k in range(3):
                w_input_hidden_updates[j][k] += delta_hidden[j] * inputs[k]
        b_output_update += delta_output

    for j in range(3):
        w_hidden_output[j] += learning_rate * (w_hidden_output_updates[j] / len(train_x))
        b_hidden[j] += learning_rate * (b_hidden_updates[j] / len(train_x))
        for k in range(3):
            w_input_hidden[j][k] += learning_rate * (w_input_hidden_updates[j][k] / len(train_x))
    b_output += learning_rate * (b_output_update / len(train_x))

    if total_error < tolerance:
        print(f"Навчання завершено на {epoch} ітерації")
        break

hidden_activations = [sigmoid(sum(test_input[k] * w_input_hidden[j][k] for k in range(3)) + b_hidden[j]) for j in range(3)]
test_prediction = sigmoid(sum(hidden_activations[j] * w_hidden_output[j] for j in range(3)) + b_output)

denormalized_prediction = denormalize(test_prediction, min_val, max_val)
print(f"Прогнозоване значення x15: {denormalized_prediction}")
