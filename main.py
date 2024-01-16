# main.py

import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
from Perceptron import Neurone
import random

def load_dataset():
    """
    Load the cat vs. non-cat dataset.

    Returns:
    - train_set_x_orig: Training dataset features.
    - train_set_y_orig: Training dataset labels.
    - test_set_x_orig: Test dataset features.
    - test_set_y_orig: Test dataset labels.
    - classes: List of classes.
    """
    with h5py.File('datasets/train_catvnoncat.h5', 'r') as train_dataset:
        train_set_x_orig = np.array(train_dataset['train_set_x'][:])
        train_set_y_orig = np.array(train_dataset['train_set_y'][:])

    with h5py.File('datasets/test_catvnoncat.h5', 'r') as test_dataset:
        test_set_x_orig = np.array(test_dataset['test_set_x'][:])
        test_set_y_orig = np.array(test_dataset['test_set_y'][:])
        classes = np.array(test_dataset['list_classes'][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# Loading dataset
x_train, y_train, x_test, y_test, classes = load_dataset()
print("Size of the training dataset: ", y_train.shape)
print("Size of the testing dataset: ", y_test.shape)
print("Size of the image: ", x_train[0].shape)
print("Shape of training dataset: ", x_train.shape)

# Preprocessing
flat_x_train = x_train.reshape(x_train.shape[0], -1)
flat_x_test = x_test.reshape(x_test.shape[0], -1)
print("flat_x_train.shape", flat_x_train.shape)
print("flat_x_test.shape", flat_x_test.shape)
train_set_x = flat_x_train / 255
test_set_x = flat_x_test / 255

# Training the perceptron
X = train_set_x
Y = y_train
i = random.randint(0, x_test.shape[0] - 1)

# Creating a Neurone instance
n_r = Neurone(X.shape[1])

# Training the model
costs, W, b = n_r.train(X, Y, epochs=2000, lr=0.1)

# Evaluating performance on the test set
perf = n_r.compute_performance(test_set_x, y_test)

# Making a prediction on a specific example
prediction = n_r.predict(train_set_x[i])

# Displaying the example image
plt.imshow(x_test[i])
plt.show()

# Displaying results
print("Performance: ", perf)
print("Prediction: ", round(float(prediction[0]), 4))

# Plotting the cost over training epochs
plt.plot(costs, np.linspace(0, len(costs) - 1, len(costs)))
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Cost over Training Epochs")
plt.show()
