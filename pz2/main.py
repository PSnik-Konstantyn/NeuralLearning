import numpy as np
import cv2
import os
import pickle

from math_functions import relu, softmax, relu_derivative

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


def compute_confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label][pred_label] += 1
    return matrix


def train_network(dataset_path, model_filename):
    X, y, classes = load_dataset(dataset_path)
    y_one_hot = np.zeros((y.size, len(classes)))
    y_one_hot[np.arange(y.size), y] = 1

    input_nodes = 1024
    hidden_nodes1 = 256
    hidden_nodes2 = 128
    output_nodes = len(classes)
    learning_rate = 0.001
    max_iterations = 2000
    batch_size = 32

    w1 = np.random.randn(input_nodes, hidden_nodes1) * np.sqrt(2 / input_nodes)
    w2 = np.random.randn(hidden_nodes1, hidden_nodes2) * np.sqrt(2 / hidden_nodes1)
    w3 = np.random.randn(hidden_nodes2, output_nodes) * np.sqrt(2 / hidden_nodes2)
    b1 = np.zeros((1, hidden_nodes1))
    b2 = np.zeros((1, hidden_nodes2))
    b3 = np.zeros((1, output_nodes))

    for epoch in range(max_iterations):
        indices = np.random.permutation(len(X))
        X, y_one_hot = X[indices], y_one_hot[indices]
        total_loss = 0

        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y_one_hot[i:i + batch_size]

            h1 = relu(np.dot(X_batch, w1) + b1)
            h2 = relu(np.dot(h1, w2) + b2)
            output = softmax(np.dot(h2, w3) + b3)

            error = output - y_batch
            total_loss += np.mean(np.sum(error ** 2, axis=1))

            d_w3 = np.dot(h2.T, error)
            d_b3 = np.sum(error, axis=0, keepdims=True)

            d_h2 = np.dot(error, w3.T) * relu_derivative(h2)
            d_w2 = np.dot(h1.T, d_h2)
            d_b2 = np.sum(d_h2, axis=0, keepdims=True)

            d_h1 = np.dot(d_h2, w2.T) * relu_derivative(h1)
            d_w1 = np.dot(X_batch.T, d_h1)
            d_b1 = np.sum(d_h1, axis=0, keepdims=True)

            w1 -= learning_rate * d_w1
            w2 -= learning_rate * d_w2
            w3 -= learning_rate * d_w3
            b1 -= learning_rate * d_b1
            b2 -= learning_rate * d_b2
            b3 -= learning_rate * d_b3

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(X):.6f}")
        # if total_loss <= 0.0004:
        #     break

    predictions = np.argmax(softmax(np.dot(relu(np.dot(relu(np.dot(X, w1) + b1), w2) + b2), w3) + b3), axis=1)
    accuracy = np.mean(predictions == y)
    print(f"Final Training Accuracy: {accuracy:.4f}")

    confusion_matrix = compute_confusion_matrix(y, predictions, len(classes))
    print("Confusion Matrix:")
    print(confusion_matrix)

    model = {'w1': w1, 'w2': w2, 'w3': w3, 'b1': b1, 'b2': b2, 'b3': b3, 'classes': classes}
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved")


def classify_image(model_filename):
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)

    file_path = input("Enter image path: ")
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32)).flatten() / 255.0

    h1 = relu(np.dot(img.reshape(1, -1), model['w1']) + model['b1'])
    h2 = relu(np.dot(h1, model['w2']) + model['b2'])
    output = softmax(np.dot(h2, model['w3']) + model['b3'])
    print(model["classes"])
    print(output)
    class_index = np.argmax(output)
    print(f'This flower is: {model["classes"][class_index]}')


if __name__ == "__main__":
    dataset_path = '/home/kostiantyn/PycharmProjects/NeuralLearning/flowers'
    model_filename = 'flower_model.pkl'

    if os.path.exists(model_filename):
        print("Model found. Skipping training...")
    else:
        print("No saved model found. Training model...")
        train_network(dataset_path, model_filename)

    classify_image(model_filename)
