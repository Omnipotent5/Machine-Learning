# Learning Linear Regression

Core question  
“How does the output change when the input changes?”

---

## Model

Single feature  
y = w·x + b

Multiple features  
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

w → feature importance (slope)  
b → bias / baseline value  

Prediction  
ŷ = Xw + b

---

## Loss functions

Loss functions define **what kind of errors the model cares about**.

### MSE (mean squared error)
- squares the error  
- large errors dominate the loss  
- very sensitive to outliers  
- smooth gradients → fast convergence  

Used when:
- outliers are rare  
- large errors are important signal  

### MAE (mean absolute error)
- absolute value of error  
- all errors treated linearly  
- robust to outliers  
- non-smooth gradients → slower convergence  

Used when:
- data is noisy  
- outliers are untrusted  

### Huber loss
- behaves like MSE for small errors  
- behaves like MAE for large errors  
- smooth + robust  

Used when:
- unsure about outliers  
- production-safe default  

---

## Gradient descent

Update rule (always the same):

w = w − α · dw  
b = b − α · db  

α  → learning rate  
dw → ∂loss / ∂w  
db → ∂loss / ∂b  

Key idea  
Loss function defines the gradients, not the update rule.

---

## Extrapolation vs interpolation

Interpolation  
Predictions made **inside** the training data range.

Extrapolation  
Predictions made **outside** the training data range.

Extrapolation assumes the learned trend continues beyond observed data  
and is inherently risky.

Loss choice affects extrapolation because it changes the learned slope.

---

## Experiments performed in code

### 1. Clean data experiment
- data generated from y = 3x + 4 + noise  
- all loss functions produce similar regression lines  
- loss values differ but geometry looks similar  

Observation  
Clean data hides loss-function differences.

---

### 2. Extreme outliers experiment
- added high-leverage outliers far from the main data  
- retrained MSE, MAE, and huber models  

Observations:
- MSE regression line shifts significantly  
- MAE remains close to the central trend  
- huber lies between MSE and MAE  

Conclusion  
Outliers expose true loss-function behavior.

---

### 3. Residual analysis

Residual = ŷ − y

- most residuals are small  
- a few extreme residuals dominate MSE  
- squared residuals cause MSE loss to grow rapidly  

Key insight  
MSE explodes through squared residuals, not visually.

---

### 4. Loss vs error visualization

Loss plotted against prediction error shows:
- MSE grows quadratically  
- MAE grows linearly  
- huber transitions from quadratic to linear  

This explains robustness differences mathematically.

---

## Practical guidelines

Few large, untrusted outliers → MAE or huber  
Few large, trusted outliers   → MSE  
Unsure about outliers         → huber  

Loss choice depends on whether outliers are signal or noise.

---

## Key takeaways

- regression lines can look similar even when losses differ greatly  
- MSE is sensitive due to squared residuals  
- MAE is robust but non-smooth  
- huber balances robustness and stability  
- differences become clearer during extrapolation  
