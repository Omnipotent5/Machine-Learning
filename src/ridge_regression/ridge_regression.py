"""
Ridge Regression from Scratch

Implements:
- L2 Regularization (Ridge penalty)
- Gradient Descent optimization
- Training loss tracking
- Feature standardization
- True vs Predicted diagnostics
- Regularization strength analysis
- Ridge coefficient path visualization

See docs/ridge_regression.md for detailed mathematical derivations.
"""

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
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent

        for _ in range(self.n_epochs):
            # Predictions
            y_predict = np.dot(X, self.weights) + self.bias
            error = y_predict - y

            # Compute Ridge Loss
            mse = np.mean(error**2)
            reg = self.alpha * np.sum(self.weights**2)
            loss = mse + reg
            self.loss_history.append(loss)

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


if __name__ == "__main__":
    # generate synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 3)
    true_w = np.array([1.5, -2.0, 0.5])
    y = X.dot(true_w) + np.random.randn(100) * 0.1

    # standardize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # train
    model = RidgeRegression(learning_rate=0.01, n_epochs=5000, alpha=0.5)
    model.fit(X, y)

    # predict
    preds = model.predict(X)

    print("Learned weights:", model.weights)
    print("True weights:", true_w)

    # Plot 1 — Loss vs Epochs
    plt.plot(model.loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE + L2)")
    plt.title("Ridge Regression Training Loss")
    plt.grid(True)
    plt.show()

    # Plot 2 — Predicted vs True Values
    plt.scatter(y, preds)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs Predicted (Ridge)")
    plt.plot([min(y), max(y)], [min(y), max(y)], "r--")
    plt.grid(True)
    plt.show()

    # Plot 3 — Effect of Alpha
    alphas = [0, 0.01, 0.1, 1, 10]
    weight_norms = []

    for a in alphas:
        model = RidgeRegression(learning_rate=0.01, n_epochs=5000, alpha=a)
        model.fit(X, y)
        weight_norms.append(np.linalg.norm(model.weights))

    plt.plot(alphas, weight_norms)
    plt.xlabel("Alpha (Regularization Strength)")
    plt.ylabel("||Weights|| (L2 Norm)")
    plt.title("Effect of Regularization on Weight Magnitude")
    plt.xscale("log")
    plt.grid(True)
    plt.show()

    # Plot 4 - each coefficient vs alpha : regularization path
    alphas = np.logspace(-2, 2, 50)
    coefs = []

    for a in alphas:
        model = RidgeRegression(learning_rate=0.01, n_epochs=5000, alpha=a)
        model.fit(X, y)
        coefs.append(model.weights)

    coefs = np.array(coefs)

    for i in range(coefs.shape[1]):
        plt.plot(alphas, coefs[:, i], label=f"W{i}")

    plt.xscale("log")
    plt.xlabel("Alpha")
    plt.ylabel("Coefficient Value")
    plt.title("Ridge Coefficient Paths")
    plt.legend()
    plt.show()

"""
Key Takeaways

• Ridge Regression adds an L2 penalty to control model complexity
• Regularization shrinks coefficients toward zero but never makes them exactly zero
• Feature scaling is critical for proper regularization behavior
• Increasing alpha increases bias and reduces variance
• Ridge stabilizes solutions when features are correlated
• Coefficient paths visualize how weights respond to regularization
• Weight norms decrease smoothly as alpha increases
• Training loss includes both data fit (MSE) and regularization penalty
• Ridge trades slight bias for improved generalization stability
"""
