import random

from first_neural import sigmoid, sigmoid_derivative


class TimeSeriesPredictor:
    def __init__(self):
        self.weights = [random.uniform(-1, 1) for _ in range(3)]
        self.learning_rate = 0.1

    def predict(self, inputs):
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs))
        return sigmoid(weighted_sum) * 10

    def train(self, training_data, epochs=10000):
        for _ in range(epochs):
            for i in range(len(training_data) - 3):
                inputs = training_data[i:i + 3]
                expected_output = training_data[i + 3]
                output = self.predict(inputs)
                error = expected_output - output
                adjustments = [self.learning_rate * error * sigmoid_derivative(output) * i for i in inputs]
                self.weights = [w + adj for w, adj in zip(self.weights, adjustments)]


training_series = [1.59, 5.73, 0.48, 5.28, 1.35, 5.91, 0.77, 5.25, 1.37, 4.42, 0.26, 4.21, 1.90, 4.08, 1.40]
predictor = TimeSeriesPredictor()
predictor.train(training_series)

test_data = training_series[-3:]
predicted_value = predictor.predict(test_data)
print(f"Predicted next value: {predicted_value}")