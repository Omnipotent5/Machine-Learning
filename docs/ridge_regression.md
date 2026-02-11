# Learning Ridge Regression

Core question  
“How do we control model complexity while learning from data?”

Ridge answers:  
“What if we penalize large weights?”

---

## Model

Single feature  
y = w·x + b  

Multiple features  
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b  

Prediction  
ŷ = Xw + b  

---

## Ridge objective

Ordinary Linear Regression minimizes:

||y − Xw||²  

Ridge modifies this to:

||y − Xw||² + λ||w||²  

λ (alpha) → regularization strength  

---

## What the regularization term does

||w||² = w₁² + w₂² + ... + wₙ²  

This term:

- penalizes large weights  
- discourages extreme slopes  
- reduces overfitting  
- improves numerical stability  

As λ increases:
- weights shrink toward zero  
- model becomes simpler  
- bias increases  
- variance decreases  

---

## Why feature scaling is important

Ridge penalizes weight magnitude.

If features are on different scales:
- larger-scale features dominate
- regularization becomes uneven

Standardization ensures:
- fair regularization
- faster convergence
- stable optimization

---

## Gradient descent

Update rule (same as linear regression):

w = w − α · dw  
b = b − α · db  

But gradients now include the regularization term.

dw = (2/n) · [Xᵀ(ŷ − y) + λw]  
db = (2/n) · Σ(ŷ − y)  

Key idea  
Regularization modifies the gradient, not the update rule.

---

## Bias-variance tradeoff

Small λ:
- low bias  
- high variance  
- risk of overfitting  

Large λ:
- higher bias  
- lower variance  
- smoother model  

Ridge intentionally trades a little bias for better generalization.

---

## Experiments performed in code

### 1. Training loss tracking

Loss = MSE + L2 penalty  

Observations:
- loss decreases rapidly at first  
- stabilizes after convergence  
- regularization contributes to final loss  

Conclusion  
Optimization is stable and convergent.

---

### 2. True vs predicted visualization

Scatter plot of:
True values vs Predicted values  

Observations:
- predictions align along diagonal  
- no systematic bias  
- ridge does not distort clean data  

Conclusion  
Regularization maintains accuracy while stabilizing weights.

---

### 3. Effect of alpha on weight magnitude

Measured ||w|| (L2 norm) for different alpha values.

Observations:
- increasing alpha reduces ||w||  
- shrinkage is smooth  
- weights never become exactly zero  

Conclusion  
Ridge shrinks coefficients but does not perform feature selection.

---

### 4. Ridge coefficient paths

Plotted each coefficient vs alpha (log scale).

Observations:
- all coefficients shrink smoothly  
- no abrupt changes  
- shrinkage depends on feature structure  

Key insight  
Ridge produces continuous shrinkage, unlike Lasso.

---

## Ridge vs Linear Regression

Linear Regression:
- minimizes only prediction error  
- may produce unstable large weights  

Ridge Regression:
- adds L2 penalty  
- controls model complexity  
- improves stability under multicollinearity  

---

## When to use Ridge

- features are correlated  
- dataset has many features  
- model shows unstable coefficients  
- want better generalization  

---

## Key takeaways

- ridge adds L2 regularization to linear regression  
- regularization shrinks weights toward zero  
- higher alpha increases bias and reduces variance  
- feature scaling is critical  
- ridge improves numerical conditioning  
- coefficients shrink smoothly but never become exactly zero  
- ridge reduces variance without inducing sparsity
