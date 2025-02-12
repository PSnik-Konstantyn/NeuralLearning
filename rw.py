import random

from functions import sigmoid, sigmoid_derivative


def normalize(data, min_val, max_val):
    return [(x - min_val) / (max_val - min_val) for x in data]


def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val


class TimeSeriesPredictor:
    def __init__(self):
        self.weights = [random.uniform(-1, 1) for _ in range(3)]
        self.learning_rate = 0.01
        self.min_val = None
        self.max_val = None

    def predict(self, inputs):
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs))
        return sigmoid(weighted_sum)

    def train(self, training_data, max_epochs=1000000, error_threshold=0.0001):
        self.min_val, self.max_val = min(training_data), max(training_data)
        normalized_data = normalize(training_data, self.min_val, self.max_val)
        prev_error = float('inf')

        for epoch in range(max_epochs):
            total_error = 0
            weight_updates = [0] * len(self.weights)
            num_updates = 0

            for i in range(len(normalized_data) - 3):
                inputs = normalized_data[i:i + 3]
                expected_output = normalized_data[i + 3]
                output = self.predict(inputs)
                error = expected_output - output
                total_error += error ** 2

                adjustments = [self.learning_rate * error * sigmoid_derivative(output) * i for i in inputs]
                weight_updates = [w + adj for w, adj in zip(weight_updates, adjustments)]
                num_updates += 1

            if num_updates > 0:
                self.weights = [w + (update / num_updates) for w, update in zip(self.weights, weight_updates)]

            if abs(total_error - prev_error) < error_threshold:
                print(epoch)
                break
            prev_error = total_error


training_series = [0.79, 3.84, 0.92, 4.50, 0.96, 5.51, 1.14, 5.32, 0.39, 4.99, 1.36, 5.81, 1.90, 4.79, 1.41]
predictor = TimeSeriesPredictor()
predictor.train(training_series)

test_data = normalize(training_series[-3:], predictor.min_val, predictor.max_val)
predicted_value = predictor.predict(test_data)
predicted_value = denormalize(predicted_value, predictor.min_val, predictor.max_val)
print(f"Predicted next value: {predicted_value}")
