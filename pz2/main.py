import random
import numpy as np
import cv2
import os
import pickle

from math_functions import sigmoid, sigmoid_derivative

def load_dataset(dataset_path):
    classes = ['daisy', 'tulip', 'rose', 'sunflower', 'dandelion']
    images, labels = [], []
    for label, flower in enumerate(classes):
        flower_path = os.path.join(dataset_path, flower)
        for img_name in os.listdir(flower_path):
            img_path = os.path.join(flower_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (32, 32)).flatten() / 255.0
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels), classes


def train_network(dataset_path, model_filename):
    X, y, classes = load_dataset(dataset_path)
    y_one_hot = np.zeros((y.size, len(classes)))
    y_one_hot[np.arange(y.size), y] = 1

    input_nodes = 1024
    hidden_nodes = 64
    output_nodes = len(classes)
    learning_rate = 0.1
    max_iterations = 1000
    tolerance = 0.0001

    w_input_hidden = np.random.uniform(-0.5, 0.5, (input_nodes, hidden_nodes))
    w_hidden_output = np.random.uniform(-0.5, 0.5, (hidden_nodes, output_nodes))
    b_hidden = np.random.uniform(-0.1, 0.1, (1, hidden_nodes))
    b_output = np.random.uniform(-0.1, 0.1, (1, output_nodes))

    for epoch in range(max_iterations):
        total_error = 0
        for i in range(len(X)):
            inputs, expected = X[i].reshape(1, -1), y_one_hot[i].reshape(1, -1)

            hidden_input = np.dot(inputs, w_input_hidden) + b_hidden
            hidden_output = sigmoid(hidden_input)
            final_input = np.dot(hidden_output, w_hidden_output) + b_output
            final_output = sigmoid(final_input)

            error = expected - final_output
            total_error += np.sum(error ** 2)

            delta_output = error * sigmoid_derivative(final_output)
            delta_hidden = np.dot(delta_output, w_hidden_output.T) * sigmoid_derivative(hidden_output)

            w_hidden_output += learning_rate * np.dot(hidden_output.T, delta_output)
            w_input_hidden += learning_rate * np.dot(inputs.T, delta_hidden)

            b_output += learning_rate * np.sum(delta_output, axis=0, keepdims=True)
            b_hidden += learning_rate * np.sum(delta_hidden, axis=0, keepdims=True)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, MSE: {total_error / len(X):.6f}")

        if total_error / len(X) < tolerance:
            print("Training complete")
            break

    model = {'w_input_hidden': w_input_hidden, 'w_hidden_output': w_hidden_output, 'b_hidden': b_hidden,
             'b_output': b_output, 'classes': classes}
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print('Model saved')


def classify_image(model_filename):
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)

    file_path = input("Enter image path: ")
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32)).flatten() / 255.0

    hidden_input = np.dot(img.reshape(1, -1), model['w_input_hidden']) + model['b_hidden']
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, model['w_hidden_output']) + model['b_output']
    final_output = sigmoid(final_input)

    class_index = np.argmax(final_output)
    print(f'This flower is: {model["classes"][class_index]}')


if __name__ == "__main__":
    dataset_path = '/NeuralLearning/flowers'
    model_filename = 'flower_model.pkl'
    print("start")
    train_network(dataset_path, model_filename)
    print("clasiffy")
    classify_image(model_filename)