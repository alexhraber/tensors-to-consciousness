# 1_automatic_differentiation.py (CuPy)
import cupy as cp

from utils import DTYPE, finite_diff_grad_scalar, finite_diff_grad_vector, viz_stage

VIZ_META = {}

print("∇ AUTOMATIC DIFFERENTIATION THEORY (CuPy via finite differences)")
print("=" * 50)

print("\n--- Chain Rule in Action ---")


def f(x):
    return cp.sin(x * x)


def analytical_derivative(x):
    return 2 * x * cp.cos(x * x)


x = cp.array(2.0, dtype=DTYPE)
f_x = f(x)
f_prime_numerical = finite_diff_grad_scalar(f, x)
f_prime_analytical = analytical_derivative(x)

print(f"f({x}) = sin({x}²) = {f_x:.6f}")
print(f"f'({x}) numerical = {f_prime_numerical:.6f}")
print(f"f'({x}) analytical = {f_prime_analytical:.6f}")
print(f"Error: {abs(f_prime_numerical - f_prime_analytical):.10f}")

viz_stage("stage_1", locals())
print("\n--- Multivariable Gradients ---")


def rosenbrock(params):
    x0, y0 = params[0], params[1]
    a, b = 1.0, 100.0
    return (a - x0) ** 2 + b * (y0 - x0**2) ** 2


point = cp.array([0.0, 0.0], dtype=DTYPE)
value = rosenbrock(point)
gradient = finite_diff_grad_vector(rosenbrock, point)

print(f"Rosenbrock at {point}: f = {value:.6f}")
print(f"Gradient: ∇f = {gradient}")
print(f"Gradient magnitude: ||∇f|| = {cp.sqrt(cp.sum(gradient**2)):.6f}")

viz_stage("stage_final", locals())
