# Perceptron.py

import numpy as np

def sigmoid(z):
    """
    Apply the sigmoid activation function.

    Parameters:
    - z: Input value.

    Returns:
    - Output value after applying the sigmoid function.
    """
    return 1 / (1 + np.exp(-z))

def linear_function(X, W, b):
    """
    Compute the linear function.

    Parameters:
    - X: Input matrix.
    - W: Weights matrix.
    - b: Bias term.

    Returns:
    - Linear output.
    """
    return np.dot(X, W) + b

class Neurone:
    """
    Logistical regression single neuron.
    """
    def __init__(self, nb_features):
        """
        Initialize a neuron with random parameters.

        Parameters:
        - nb_features: Number of input features.
        """
        self.W, self.b = self.init_params(nb_features)

    def forward(self, X, Y):
        """
        Perform forward propagation.

        Parameters:
        - X: Input matrix.
        - Y: Target values.

        Returns:
        - cost: Computed cost.
        - Y_hat: Predicted values.
        """
        W = self.W
        b = self.b
        Z = linear_function(X, W, b)
        Y_hat = sigmoid(Z)
        m = X.shape[1]
        cost = -1/m * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
        return cost, Y_hat

    def backward(self, Y, Y_hat, X, learning_rate):
        """
        Perform backward propagation and update parameters.

        Parameters:
        - Y: Target values.
        - Y_hat: Predicted values.
        - X: Input matrix.
        - learning_rate: Learning rate for gradient descent.

        Returns:
        - Updated weights (W) and bias (b).
        """
        W = self.W
        b = self.b
        m = X.shape[1]
        d_W = 1/m * (np.dot(X.T, (Y_hat - Y)))
        d_b = 1/m * np.sum(Y_hat - Y)
        n_W = W - learning_rate * d_W
        n_b = b - learning_rate * d_b
        return n_W, n_b

    def init_params(self, dim):
        """
        Initialize weights (W) and bias (b) parameters.

        Parameters:
        - dim: Number of input features.

        Returns:
        - W: Initialized weights.
        - b: Initialized bias.
        """
        W = np.zeros((dim, 1))
        b = 0
        return W, b

    def compute_performance(self, X, Y_):
        """
        Compute the performance of the model.

        Parameters:
        - X: Input matrix.
        - Y_: Target values.

        Returns:
        - Percentage of correct predictions.
        """
        Y = Y_.T
        W = self.W
        b = self.b
        Z = linear_function(X, W, b)
        Y_hat = sigmoid(Z)
        correct_prediction_tab = (Y_hat > 0.5) == Y
        correct_prediction_count = np.count_nonzero(correct_prediction_tab == True)
        percentage = (correct_prediction_count * 100) / correct_prediction_tab.size
        return percentage

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        - X: Input matrix.

        Returns:
        - Predicted values.
        """
        Z = linear_function(X, self.W, self.b)
        Y_hat = sigmoid(Z)
        return Y_hat

    def train(self, X, Y_, epochs, lr):
        """
        Train the model using gradient descent.

        Parameters:
        - X: Input matrix.
        - Y_: Target values.
        - epochs: Number of training epochs.
        - lr: Learning rate for gradient descent.

        Returns:
        - costs: List of costs over training.
        - W: Trained weights.
        - b: Trained bias.
        """
        Y = Y_.T
        costs = []
        for i in range(0, epochs):
            cost = 0
            W = self.W
            b = self.b
            cost, Y_hat = self.forward(X, Y)
            costs.append(cost)
            self.W, self.b = self.backward(Y, Y_hat, X, lr)
        return costs, W, b
