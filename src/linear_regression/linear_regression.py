"""
Linear Regression from scratch

Implements:
- MSE, MAE, Huber loss
- Gradient descent
- Outlier and extrapolation experiments

See docs/linear_regression.md for detailed notes.
"""

import matplotlib.pyplot as plt
import numpy as np


class LinearRegression:
    def __init__(self, learning_rate, n_epochs=1000, loss="mse", delta=1.0):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.loss = loss  # "mse", "mae", "huber"
        self.delta = delta  # used only for Huber
        self.weights = None
        self.bias = 0

    def fit(self, X, y):

        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_epochs):
            # Forward pass
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

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


class LossStudy:
    def __init__(self, delta=1.0):
        self.delta = delta

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mae(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def huber(self, y_true, y_pred):
        error = y_true - y_pred
        mask = np.abs(error) <= self.delta

        squared = 0.5 * error**2
        linear = self.delta * (np.abs(error) - 0.5 * self.delta)

        return np.mean(np.where(mask, squared, linear))

    def plot_loss_vs_error(self, error_range=(-5, 5)):
        errors = np.linspace(error_range[0], error_range[1], 400)

        mse_loss = errors**2
        mae_loss = np.abs(errors)
        huber_loss = np.where(
            np.abs(errors) <= self.delta,
            0.5 * errors**2,
            self.delta * (np.abs(errors) - 0.5 * self.delta),
        )

        plt.plot(errors, mse_loss, label="MSE Loss")
        plt.plot(errors, mae_loss, label="MAE Loss")
        plt.plot(errors, huber_loss, label="Huber Loss")

        plt.xlabel("Prediction Error (ŷ − y)")
        plt.ylabel("Loss")
        plt.title("Loss Functions vs Prediction Error")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Generate clean data

    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 3 * X.flatten() + 4 + np.random.randn(100)

    # Generate data with extreme outliers
    X_outliers = np.array([[3.5], [4.0], [4.5]])
    y_outliers = np.array([40, -30, 50])

    X = np.vstack((X, X_outliers))
    y = np.concatenate((y, y_outliers))

    # Create and Train Model

    # MSE
    model_mse = LinearRegression(learning_rate=0.01, loss="mse")
    model_mse.fit(X, y)

    # MAE
    model_mae = LinearRegression(learning_rate=0.01, loss="mae")
    model_mae.fit(X, y)

    # Huber
    model_huber = LinearRegression(learning_rate=0.01, loss="huber", delta=0.5)
    model_huber.fit(X, y)

    # Predictions below are extrapolation
    # (values outside the training data range [0, 2])

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

    # Create line space
    X_line = np.linspace(0, 2, 100).reshape(-1, 1)

    y_mse = model_mse.predict(X_line)
    y_mae = model_mae.predict(X_line)
    y_huber = model_huber.predict(X_line)

    # Plot
    plt.scatter(X, y)
    plt.plot(X_line, y_mse, label="MSE Regression")
    plt.plot(X_line, y_mae, label="MAE Regression")
    plt.plot(X_line, y_huber, label="Huber Regression")

    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression with Different Loss Functions")
    plt.legend()
    plt.show()

    # Study Loss Function

    loss_study = LossStudy(delta=1.0)

    print("\nLoss comparison on MSE model predictions:")
    print("MSE:", loss_study.mse(y, model_mse.predict(X)))
    print("MAE:", loss_study.mae(y, model_mse.predict(X)))
    print("Huber:", loss_study.huber(y, model_mse.predict(X)))

    # Plot loss vs error behavior
    loss_study.plot_loss_vs_error()

    # Residual analysis (to visualize MSE sensitivity to outliers)
    residuals = model_mse.predict(X) - y

    plt.scatter(X.flatten(), residuals)
    plt.axhline(0)
    plt.title("Residuals of MSE Model")
    plt.xlabel("X")
    plt.ylabel("Residual (ŷ − y)")
    plt.show()


"""
Key Takeaways

• Regression lines may look similar even when losses differ greatly
• MSE explodes through squared residuals, not visually
• MAE is robust but non-smooth
• Huber balances robustness and stability
• Loss choice depends on whether outliers are signal or noise
"""
