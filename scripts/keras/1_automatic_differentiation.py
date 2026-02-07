# 1_automatic_differentiation.py (Keras)
import numpy as np

from utils import DTYPE, finite_diff_grad_scalar, finite_diff_grad_vector

print("∇ AUTOMATIC DIFFERENTIATION THEORY (Keras, numerical gradients)")
print("=" * 50)

print("\n--- Chain Rule in Action ---")


def f(x):
    return np.sin(x * x)


def analytical_derivative(x):
    return 2 * x * np.cos(x * x)


x = np.array(2.0, dtype=np.float32)
f_x = f(x)
f_prime_numerical = finite_diff_grad_scalar(f, x)
f_prime_analytical = analytical_derivative(x)

print(f"f({x}) = sin({x}²) = {f_x:.6f}")
print(f"f'({x}) numerical = {f_prime_numerical:.6f}")
print(f"f'({x}) analytical = {f_prime_analytical:.6f}")
print(f"Error: {abs(f_prime_numerical - f_prime_analytical):.10f}")

print("\n--- Multivariable Gradients ---")


def rosenbrock(params):
    x0, y0 = params[0], params[1]
    a, b = 1.0, 100.0
    return (a - x0) ** 2 + b * (y0 - x0**2) ** 2


point = np.array([0.0, 0.0], dtype=np.float32)
value = rosenbrock(point)
gradient = finite_diff_grad_vector(rosenbrock, point)

print(f"Rosenbrock at {point}: f = {value:.6f}")
print(f"Gradient: ∇f = {gradient}")
print(f"Gradient magnitude: ||∇f|| = {np.sqrt(np.sum(gradient**2)):.6f}")
