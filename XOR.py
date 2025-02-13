import random

from math_functions import sigmoid, sigmoid_derivative

class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)

    def feedforward(self, inputs):
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return sigmoid(weighted_sum)

    def adjust(self, inputs, error, learning_rate):
        adjustments = [learning_rate * error * sigmoid_derivative(self.feedforward(inputs)) * i for i in inputs]
        self.weights = [w + adj for w, adj in zip(self.weights, adjustments)]
        self.bias += learning_rate * error * sigmoid_derivative(self.feedforward(inputs))


class MLP:
    def __init__(self, num_inputs, num_hidden):
        self.hidden_layer = [Neuron(num_inputs) for _ in range(num_hidden)]
        self.output_neuron = Neuron(num_hidden)
        self.learning_rate = 0.1

    def feedforward(self, inputs):
        hidden_outputs = [neuron.feedforward(inputs) for neuron in self.hidden_layer]
        return self.output_neuron.feedforward(hidden_outputs)

    def train(self, inputs, expected_output):
        hidden_outputs = [neuron.feedforward(inputs) for neuron in self.hidden_layer]
        output = self.output_neuron.feedforward(hidden_outputs)
        output_error = expected_output - output

        self.output_neuron.adjust(hidden_outputs, output_error, self.learning_rate)
        for i, neuron in enumerate(self.hidden_layer):
            hidden_error = output_error * self.output_neuron.weights[i]
            neuron.adjust(inputs, hidden_error, self.learning_rate)


training_data = {
    "XOR": [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
}


def train_xor():
    mlp = MLP(num_inputs=2, num_hidden=2)
    for _ in range(10000):
        for inputs, expected in training_data["XOR"]:
            mlp.train(inputs, expected)
    return mlp


def test_xor(mlp):
    print("Testing XOR function:")
    for inputs, expected in training_data["XOR"]:
        output = mlp.feedforward(inputs)
        print(f"Input: {inputs} -> Output: {round(output)} (Expected: {expected})")


mlp_xor = train_xor()
test_xor(mlp_xor)
