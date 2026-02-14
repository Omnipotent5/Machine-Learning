# Learning Linear Regression

Core question  
**"How does the output change when the input changes?"**

---

## Model

### Single feature

y = w·x + b  

### Multiple features

y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b  

- w → feature importance (slope)  
- b → bias / baseline value  

### Prediction

ŷ = Xw + b  

---

## Loss Functions

Loss functions define **what kind of errors the model cares about**.

### Mean Squared Error (MSE)

\[
E = \frac{1}{n} \sum_{i=1}^{n} (y_i - (m x_i + b))^2
\]

- squares the error  
- large errors dominate the loss  
- very sensitive to outliers  
- smooth gradients → fast convergence  

Used when:
- outliers are rare  
- large errors are important signal  

---

### Mean Absolute Error (MAE)

\[
E = \frac{1}{n} \sum_{i=1}^{n} |y_i - (m x_i + b)|
\]

- absolute value of error  
- all errors treated linearly  
- robust to outliers  
- non-smooth gradients → slower convergence  

Used when:
- data is noisy  
- outliers are untrusted  

---

### Huber Loss

For error \( e \):

\[
L_\delta(e) =
\begin{cases}
\frac{1}{2} e^2 & \text{if } |e| \le \delta \\
\delta(|e| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
\]

- behaves like MSE for small errors  
- behaves like MAE for large errors  
- smooth + robust  

Used when:
- unsure about outliers  
- production-safe default  

---

## Gradient Descent

### Update Rule (always the same)

\[
w = w - \alpha \cdot dw
\]

\[
b = b - \alpha \cdot db
\]

Where:

- α  → learning rate  
- dw → ∂loss / ∂w  
- db → ∂loss / ∂b  

**Key idea:**  
The loss function defines the gradients, not the update rule.

---

## Gradient Derivation for MSE

Starting from:

\[
E = \frac{1}{n} \sum_{i=1}^{n} (y_i - (m x_i + b))^2
\]

### Partial derivative with respect to m

\[
\frac{\partial E}{\partial m}
=
\frac{1}{n}
\sum_{i=1}^{n}
2 (y_i - (m x_i + b)) (-x_i)
\]

\[
=
-\frac{2}{n}
\sum_{i=1}^{n}
x_i (y_i - (m x_i + b))
\]

---

### Partial derivative with respect to b

\[
\frac{\partial E}{\partial b}
=
-\frac{2}{n}
\sum_{i=1}^{n}
(y_i - (m x_i + b))
\]

---

### Final Gradient Update (Single Feature)

\[
m = m - \alpha \left( -\frac{2}{n}
\sum_{i=1}^{n}
x_i (y_i - (m x_i + b)) \right)
\]

\[
b = b - \alpha \left( -\frac{2}{n}
\sum_{i=1}^{n}
(y_i - (m x_i + b)) \right)
\]

---

## Extrapolation vs Interpolation

### Interpolation
Predictions made **inside** the training data range.

### Extrapolation
Predictions made **outside** the training data range.

Extrapolation assumes the learned trend continues beyond observed data  
and is inherently risky.

Loss choice affects extrapolation because it changes the learned slope.

---

## Experiments Performed in Code

### 1. Clean Data Experiment

- data generated from y = 3x + 4 + noise  
- all loss functions produce similar regression lines  
- loss values differ but geometry looks similar  

**Observation:**  
Clean data hides loss-function differences.

---

### 2. Extreme Outliers Experiment

- added high-leverage outliers far from the main data  
- retrained MSE, MAE, and Huber models  

**Observations:**

- MSE regression line shifts significantly  
- MAE remains close to the central trend  
- Huber lies between MSE and MAE  

**Conclusion:**  
Outliers expose true loss-function behavior.

---

### 3. Residual Analysis

Residual = ŷ − y

- most residuals are small  
- a few extreme residuals dominate MSE  
- squared residuals cause MSE loss to grow rapidly  

**Key insight:**  
MSE explodes through squared residuals, not visually.

---

### 4. Loss vs Error Visualization

Loss plotted against prediction error shows:

- MSE grows quadratically  
- MAE grows linearly  
- Huber transitions from quadratic to linear  

This explains robustness differences mathematically.

---

## Practical Guidelines

- Few large, untrusted outliers → MAE or Huber  
- Few large, trusted outliers → MSE  
- Unsure about outliers → Huber  

Loss choice depends on whether outliers are signal or noise.

---

## Key Takeaways

- regression lines can look similar even when losses differ greatly  
- MSE is sensitive due to squared residuals  
- MAE is robust but non-smooth  
- Huber balances robustness and stability  
- differences become clearer during extrapolation  
