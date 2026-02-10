# Learning Linear Regression

# “How does the output change when the input changes?”

# For one feature: y = wx + b
# For multiple features: y = w1x1 + w2x2 + ... + wnxn + b
# w - importance/ influence of a feature
# b - starting value

# Prediction Formula : ŷ = Xw + b
# Loss Function : MSE, MAE, HUBER
# MSE   → squares error → punishes outliers heavily
# MAE   → absolute error → robust to outliers
# Huber → MSE for small errors, MAE for large errors
# Gradient Descent :
# w = w − α · dw
# b = b − α · db
# α : learning rate
# dw : ∂(Loss)/∂w
# db = ∂(Loss)/∂b
# Gradient Descent rule → ALWAYS same
# Loss function → defines gradients
# Gradient descent update equations stay the same for all loss functions.
# Only the gradients change based on how the loss penalizes errors.

import numpy as np


class LinearRegression:
    def __init__(self, learning_rate, n_epochs=1000, loss="mse", delta=1.0):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.loss = loss
        self.delta = delta
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_epochs):
            y_predicted = np.dot(X, self.weights) + self.bias
            error = y_predicted - y

            # Calculate Gradients

            if self.loss == "mse":
                dw = (1 / n_samples) * np.dot(X.T, error)
                db = (1 / n_samples) * np.sum(error)

            elif self.loss == "mae":
                sign_error = np.sign(error)
                dw = (1 / n_samples) * np.dot(X.T, sign_error)
                db = (1 / n_samples) * np.sum(sign_error)

            elif self.loss == "huber":
                mask = np.abs(error) <= self.delta
                grad = np.where(mask, error, self.delta * np.sign(error))

                dw = (1 / n_samples) * np.dot(X.T, grad)
                db = (1 / n_samples) * np.sum(grad)

            else:
                raise ValueError("Loss must be 'mse', 'mae', or 'huber'")

            # Update Parameters

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Calculate Loss

            loss = np.mean((error) ** 2)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


if __name__ == "__main__":
    # Generate sample data

    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 3 * X.flatten() + 4 + np.random.randn(100)

    # Create and Train Model

    # MSE
    model_mse = LinearRegression(learning_rate=0.01, loss="mse")
    model_mse.fit(X, y)

    # MAE
    model_mae = LinearRegression(learning_rate=0.01, loss="mae")
    model_mae.fit(X, y)

    # Huber
    model_huber = LinearRegression(learning_rate=0.01, loss="huber", delta=1.0)
    model_huber.fit(X, y)

    # Make Predictions

    predictions1 = model_mse.predict([[5], [10], [15], [20]])
    predictions2 = model_mae.predict([[5], [10], [15], [20]])
    predictions3 = model_huber.predict([[5], [10], [15], [20]])

    # Print Results
    print("For Mean Squared Error \n")
    print("Predictions: ", predictions1)
    print("Learned Weights: ", model_mse.weights)
    print("Learned Bias: ", model_mse.bias)
    print("\n")

    print("For Mean Absolute Error \n")
    print("Predictions: ", predictions2)
    print("Learned Weights: ", model_mae.weights)
    print("Learned Bias: ", model_mae.bias)
    print("\n")

    print("For Huber Error \n")
    print("Predictions: ", predictions3)
    print("Learned Weights: ", model_huber.weights)
    print("Learned Bias: ", model_huber.bias)

# MSE produces parameters influenced by outliers,
# MAE fits the median trend and is more robust, and
# Huber provides a balance by behaving like MSE for small errors and MAE for large errors.
