# 2_optimization_theory.py
import mlx.core as mx
from utils import viz_stage

VIZ_META = {}

print("🎯 OPTIMIZATION THEORY")
print("=" * 50)

# Set global dtype for consistency
DTYPE = mx.float32

# 1. GRADIENT DESCENT: Following the negative gradient
print("\n--- Gradient Descent Dynamics ---")

def quadratic_bowl(x):
    """Simple quadratic: f(x) = x² + 0.1x + 5"""
    return x**2 + 0.1*x + 5

def gradient_descent_step(x, learning_rate):
    """One step of gradient descent"""
    grad = mx.grad(quadratic_bowl)(x)
    return x - learning_rate * grad

# Simulate gradient descent
x = mx.array(10.0, dtype=DTYPE)  # Start far from minimum
learning_rate = 0.1
history = [float(x)]

print(f"Starting at x = {x:.6f}, f(x) = {quadratic_bowl(x):.6f}")

for i in range(20):
    x = gradient_descent_step(x, learning_rate)
    history.append(float(x))
    if i % 5 == 0:
        print(f"Step {i+1}: x = {x:.6f}, f(x) = {quadratic_bowl(x):.6f}")

# 2. MOMENTUM: Physics-inspired optimization
viz_stage("stage_1", locals())
print("\n--- Momentum Methods ---")

def gradient_descent_with_momentum(func, x_start, lr, momentum, steps):
    """Gradient descent with momentum"""
    x = x_start
    velocity = mx.array(0.0, dtype=DTYPE)
    history = [float(x)]
    
    for i in range(steps):
        grad = mx.grad(func)(x)
        velocity = momentum * velocity + lr * grad
        x = x - velocity
        history.append(float(x))
    
    return x, history

# Compare standard GD vs momentum
final_vanilla, hist_vanilla = gradient_descent_with_momentum(
    quadratic_bowl, mx.array(10.0, dtype=DTYPE), 0.05, 0.0, 50)
final_momentum, hist_momentum = gradient_descent_with_momentum(
    quadratic_bowl, mx.array(10.0, dtype=DTYPE), 0.05, 0.9, 50)

print(f"Vanilla GD final: x = {final_vanilla:.6f}")
print(f"Momentum GD final: x = {final_momentum:.6f}")

# 3. ADAPTIVE LEARNING RATES: Adam concept
viz_stage("stage_2", locals())
print("\n--- Adaptive Optimization (Adam concept) ---")

def simple_adam_step(x, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """Simplified Adam update"""
    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * grad
    # Update biased second raw moment estimate  
    v = beta2 * v + (1 - beta2) * grad**2
    
    # Compute bias-corrected first moment estimate
    m_hat = m / (1 - beta1**t)
    # Compute bias-corrected second raw moment estimate
    v_hat = v / (1 - beta2**t)
    
    # Update parameters
    x = x - lr * m_hat / (mx.sqrt(v_hat) + eps)
    
    return x, m, v

# Demo adaptive learning
x = mx.array(10.0, dtype=DTYPE)
m = mx.array(0.0, dtype=DTYPE)
v = mx.array(0.0, dtype=DTYPE)

print("Adam-style optimization:")
for t in range(1, 11):
    grad = mx.grad(quadratic_bowl)(x)
    x, m, v = simple_adam_step(x, grad, m, v, t)
    if t % 2 == 0:
        print(f"Step {t}: x = {x:.6f}, f(x) = {quadratic_bowl(x):.6f}")

viz_stage("stage_final", locals())
