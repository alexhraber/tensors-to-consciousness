# 2_optimization_theory.py (CuPy)
import cupy as cp

from utils import DTYPE, finite_diff_grad_scalar

print("🎯 OPTIMIZATION THEORY (CuPy)")
print("=" * 50)

print("\n--- Gradient Descent Dynamics ---")


def quadratic_bowl(x):
    return x**2 + 0.1 * x + 5


def gradient_descent_step(x, learning_rate):
    grad = finite_diff_grad_scalar(quadratic_bowl, x)
    return x - learning_rate * grad


x = cp.array(10.0, dtype=DTYPE)
learning_rate = 0.1
history = [float(x)]

print(f"Starting at x = {x:.6f}, f(x) = {quadratic_bowl(x):.6f}")
for i in range(20):
    x = gradient_descent_step(x, learning_rate)
    history.append(float(x))
    if i % 5 == 0:
        print(f"Step {i + 1}: x = {x:.6f}, f(x) = {quadratic_bowl(x):.6f}")

print("\n--- Momentum Methods ---")


def gradient_descent_with_momentum(func, x_start, lr, momentum, steps):
    x0 = x_start
    velocity = cp.array(0.0, dtype=DTYPE)
    hist = [float(x0)]

    for _ in range(steps):
        grad = finite_diff_grad_scalar(func, x0)
        velocity = momentum * velocity + lr * grad
        x0 = x0 - velocity
        hist.append(float(x0))

    return x0, hist


final_vanilla, hist_vanilla = gradient_descent_with_momentum(
    quadratic_bowl, cp.array(10.0, dtype=DTYPE), 0.05, 0.0, 50
)
final_momentum, hist_momentum = gradient_descent_with_momentum(
    quadratic_bowl, cp.array(10.0, dtype=DTYPE), 0.05, 0.9, 50
)

print(f"Vanilla GD final: x = {final_vanilla:.6f}")
print(f"Momentum GD final: x = {final_momentum:.6f}")

print("\n--- Adaptive Optimization (Adam concept) ---")


def simple_adam_step(x0, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    x0 = x0 - lr * m_hat / (cp.sqrt(v_hat) + eps)
    return x0, m, v


x = cp.array(10.0, dtype=DTYPE)
m = cp.array(0.0, dtype=DTYPE)
v = cp.array(0.0, dtype=DTYPE)

print("Adam-style optimization:")
for t in range(1, 11):
    grad = finite_diff_grad_scalar(quadratic_bowl, x)
    x, m, v = simple_adam_step(x, grad, m, v, t)
    if t % 2 == 0:
        print(f"Step {t}: x = {x:.6f}, f(x) = {quadratic_bowl(x):.6f}")
