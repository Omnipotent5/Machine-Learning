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

Lδ(e) =  
  ½e²    if |e| ≤ δ  
  δ(|e| − ½δ) otherwise  

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

∂E/∂m  
= (1/n) Σᵢ₌₁ⁿ 2(yᵢ − (m·xᵢ + b))(−xᵢ)  

= −(2/n) Σᵢ₌₁ⁿ xᵢ (yᵢ − (m·xᵢ + b))

---

### Partial derivative with respect to b

∂E/∂b  
= −(2/n) Σᵢ₌₁ⁿ (yᵢ − (m·xᵢ + b))

---

### Final Gradient Update (Single Feature)

m = m − α [ −(2/n) Σᵢ₌₁ⁿ xᵢ (yᵢ − (m·xᵢ + b)) ]  

b = b − α [ −(2/n) Σᵢ₌₁ⁿ (yᵢ − (m·xᵢ + b)) ]
