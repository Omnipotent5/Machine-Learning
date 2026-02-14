## Loss Functions

Loss functions define **what kind of errors the model cares about**.

### Mean Squared Error (MSE)

E = (1/n) Σᵢ₌₁ⁿ (yᵢ − (m·xᵢ + b))²

- squares the error  
- large errors dominate the loss  
- very sensitive to outliers  
- smooth gradients → fast convergence  

Used when:
- outliers are rare  
- large errors are important signal  

---

### Mean Absolute Error (MAE)

E = (1/n) Σᵢ₌₁ⁿ |yᵢ − (m·xᵢ + b)|

- absolute value of error  
- all errors treated linearly  
- robust to outliers  
- non-smooth gradients → slower convergence  

Used when:
- data is noisy  
- outliers are untrusted  

---

### Huber Loss

For error e:

Lδ(e) =  ½e²            if |e| ≤ δ  
         δ(|e| − ½δ)    otherwise  

- behaves like MSE for small errors  
- behaves like MAE for large errors  
- smooth + robust  

Used when:
- unsure about outliers  
- production-safe default  

---

## Gradient Descent

### Update Rule

w = w − α·dw  
b = b − α·db  

Where:

α  → learning rate  
dw → ∂E/∂w  
db → ∂E/∂b  

**Key idea:**  
The loss function defines the gradients, not the update rule.

---

## Gradient Derivation for MSE

Starting from:

E = (1/n) Σᵢ₌₁ⁿ (yᵢ − (m·xᵢ + b))²

---

### Partial derivative with respect to m

∂E/∂m   = (1/n) Σᵢ₌₁ⁿ 2(yᵢ − (m·xᵢ + b))(−xᵢ)  

        = −(2/n) Σᵢ₌₁ⁿ xᵢ (yᵢ − (m·xᵢ + b))

---

### Partial derivative with respect to b

∂E/∂b  = −(2/n) Σᵢ₌₁ⁿ (yᵢ − (m·xᵢ + b))

---

### Final Gradient Update (Single Feature)

m = m − α [ −(2/n) Σᵢ₌₁ⁿ xᵢ (yᵢ − (m·xᵢ + b)) ]  

b = b − α [ −(2/n) Σᵢ₌₁ⁿ (yᵢ − (m·xᵢ + b)) ]

# Matrix Form of Linear Regression (Normal Equation)

---

## Residual Sum of Squares (RSS)

RSS = Σᵢ₌₁ⁿ (yᵢ − ŷᵢ)²  

Since:

ŷᵢ = xᵢᵀ β  

We can write:

RSS(β) = (y − Xβ)ᵀ (y − Xβ)

---

## Expanding RSS

RSS(β)  
= yᵀy − βᵀXᵀy − yᵀXβ + βᵀXᵀXβ  

Because βᵀXᵀy and yᵀXβ are scalars (equal), we combine:

RSS(β)  
= yᵀy − 2βᵀXᵀy + βᵀXᵀXβ

---

## Taking Derivative with Respect to β

∂RSS / ∂β  
= 0 − 2Xᵀy + 2XᵀXβ  

Set derivative equal to zero:

−2Xᵀy + 2XᵀXβ = 0  

---

## Solving for β

2Xᵀy = 2XᵀXβ  

Divide by 2:

Xᵀy = XᵀXβ  

Multiply both sides by (XᵀX)⁻¹:

β̂ = (XᵀX)⁻¹ Xᵀy  

This is the **Normal Equation**.

---

# Prediction in Matrix Form

ŷ = Xβ̂  

---

# Understanding the Design Matrix

To include bias (intercept):

Let x₀ = 1  

Then each training example becomes:

xᵢ = [ 1, x₁, x₂, x₃, ... ]

And parameter vector:

β = [ b, w₁, w₂, w₃, ... ]ᵀ

Where:

- b  → intercept  
- wⱼ → feature coefficients  

---

# Scalar Version (Single Feature Case)

For single feature:

RSS = Σᵢ₌₁ⁿ (yᵢ − (mxᵢ + b))²  

Which is exactly the form used in the gradient descent implementation.

---

# Key Insight

Gradient Descent:
- Iterative optimization
- Works for large datasets
- Does not require matrix inversion

Normal Equation:
- Closed-form analytical solution
- Requires computing (XᵀX)⁻¹
- Expensive for high-dimensional data

