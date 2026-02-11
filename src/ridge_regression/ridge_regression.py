import matplotlib.pyplot as plt
import numpy as np


class RidgeRegression:
    def __init__(self, learning_rate=0.01, n_epochs=1000, alpha=0.1):
        """
        n_epochs : number of gradient steps
        alpha    : regularization strength (lambda)
        """

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent

        for _ in range(self.n_epochs):
            # Predictions
            y_predict = np.dot(X, self.weights) + self.bias

            # Compute Gradients
            dw = (2 / n_samples) * (
                np.dot(X.T, (y_predict - y)) + self.alpha * self.weights
            )
            db = (2 / n_samples) * np.sum(y_predict - y)

            # Update Parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
