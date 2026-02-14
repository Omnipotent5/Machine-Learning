"""
Linear Regression from scratch (MSE version)

Implements:
- Mean Squared Error (MSE)
- Batch Gradient Descent
- Single feature regression (YearsExperience → Salary)

This script:
- Loads salary dataset
- Trains linear regression using MSE
- Prints learned slope (m) and intercept (b)
- Plots regression line against training data

See docs/linear_regression.md for full theory and derivations.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("src/linear_regression/archive/Salary_dataset.csv")
plt.scatter(data["YearsExperience"], data["Salary"])
plt.show()

def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i]["YearsExperience"]
        y = points.iloc[i]["Salary"]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

def gradient_descent(m_now, b_now, points, learning_rate):
    m_gradient = 0
    b_gradient = 0
    n = len(points)
    
    for i in range(n):
        x = points.iloc[i]["YearsExperience"]
        y = points.iloc[i]["Salary"]
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
    
    m = m_now - (learning_rate * m_gradient)
    b = b_now - (learning_rate * b_gradient)
    
    return m, b
        
learning_rate = 0.0001
m = 0
b = 0
epochs = 1000

for i in range(epochs):
    if i % 100 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, data, learning_rate)

print(m, b)
plt.scatter(data["YearsExperience"], data["Salary"], color="black")
plt.plot(data["YearsExperience"], m * data["YearsExperience"] + b, color="red")
plt.show()

"""
Key Observations (MSE Version)

• Model minimizes squared residuals
• Large errors have disproportionately large impact
• Learned slope represents change in salary per year of experience
• Gradient descent iteratively updates parameters
• Convergence depends on learning rate selection
"""

