import random

from functions import sigmoid_derivative, sigmoid


class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
        self.learning_rate = 0.5

    def feedforward(self, inputs):
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return sigmoid(weighted_sum)

    def train(self, inputs, expected_output):
        output = self.feedforward(inputs)
        error = expected_output - output
        adjustments = [self.learning_rate * error * sigmoid_derivative(output) * i for i in inputs]
        self.weights = [w + adj for w, adj in zip(self.weights, adjustments)]
        self.bias += self.learning_rate * error * sigmoid_derivative(output)

training_data = {
    "AND": [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)],
    "OR": [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)],
    "NOT": [([0], 1), ([1], 0)],
    "XOR": [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)],
    "CUSTOM": [([0, 0, 0], 1), ([0, 1, 0], 1), ([1, 0, 0], 0), ([1, 1, 1], 1)]
}

def train_logical_function(func_name):
    num_inputs = len(training_data[func_name][0][0])
    neuron = Neuron(num_inputs)
    for _ in range(10000):
        for inputs, expected in training_data[func_name]:
            neuron.train(inputs, expected)
    return neuron

def test_logical_function(neuron, func_name):
    print(f"Testing {func_name} function:")
    for inputs, expected in training_data[func_name]:
        output = neuron.feedforward(inputs)
        print(f"Input: {inputs} -> Output: {round(output)} (Expected: {expected})")

for func in training_data:
    trained_neuron = train_logical_function(func)
    test_logical_function(trained_neuron, func)