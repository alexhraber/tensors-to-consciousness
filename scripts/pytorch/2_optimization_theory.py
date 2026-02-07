# 2_optimization_theory.py (PyTorch)
import torch

from utils import DTYPE, scalar

print("🎯 OPTIMIZATION THEORY (PyTorch)")
print("=" * 50)

print("\n--- Gradient Descent Dynamics ---")


def quadratic_bowl(x):
    return x**2 + 0.1 * x + 5


def gradient_descent_step(x, learning_rate):
    x_var = x.clone().detach().requires_grad_(True)
    loss = quadratic_bowl(x_var)
    loss.backward()
    return (x_var - learning_rate * x_var.grad).detach()


x = torch.tensor(10.0, dtype=DTYPE)
learning_rate = 0.1
history = [scalar(x)]

print(f"Starting at x = {scalar(x):.6f}, f(x) = {scalar(quadratic_bowl(x)):.6f}")
for i in range(20):
    x = gradient_descent_step(x, learning_rate)
    history.append(scalar(x))
    if i % 5 == 0:
        print(f"Step {i + 1}: x = {scalar(x):.6f}, f(x) = {scalar(quadratic_bowl(x)):.6f}")

print("\n--- Momentum Methods ---")


def gradient_descent_with_momentum(func, x_start, lr, momentum, steps):
    x0 = x_start.clone().detach()
    velocity = torch.tensor(0.0, dtype=DTYPE)
    hist = [scalar(x0)]

    for _ in range(steps):
        x_var = x0.clone().detach().requires_grad_(True)
        loss = func(x_var)
        loss.backward()
        grad = x_var.grad.detach()

        velocity = momentum * velocity + lr * grad
        x0 = (x_var - velocity).detach()
        hist.append(scalar(x0))

    return x0, hist


final_vanilla, hist_vanilla = gradient_descent_with_momentum(
    quadratic_bowl, torch.tensor(10.0, dtype=DTYPE), 0.05, 0.0, 50
)
final_momentum, hist_momentum = gradient_descent_with_momentum(
    quadratic_bowl, torch.tensor(10.0, dtype=DTYPE), 0.05, 0.9, 50
)

print(f"Vanilla GD final: x = {scalar(final_vanilla):.6f}")
print(f"Momentum GD final: x = {scalar(final_momentum):.6f}")

print("\n--- Adaptive Optimization (Adam concept) ---")


def simple_adam_step(x0, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    x0 = x0 - lr * m_hat / (torch.sqrt(v_hat) + eps)
    return x0, m, v


x = torch.tensor(10.0, dtype=DTYPE)
m = torch.tensor(0.0, dtype=DTYPE)
v = torch.tensor(0.0, dtype=DTYPE)

print("Adam-style optimization:")
for t in range(1, 11):
    x_var = x.clone().detach().requires_grad_(True)
    loss = quadratic_bowl(x_var)
    loss.backward()
    grad = x_var.grad.detach()

    x, m, v = simple_adam_step(x_var.detach(), grad, m, v, t)
    if t % 2 == 0:
        print(f"Step {t}: x = {scalar(x):.6f}, f(x) = {scalar(quadratic_bowl(x)):.6f}")
