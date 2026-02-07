# 1_automatic_differentiation.py (JAX)
import jax
import jax.numpy as jnp

from utils import DTYPE, viz_stage

VIZ_META = {}

print("∇ AUTOMATIC DIFFERENTIATION THEORY (JAX)")
print("=" * 50)

print("\n--- Chain Rule in Action ---")


def f(x):
    return jnp.sin(x * x)


def analytical_derivative(x):
    return 2 * x * jnp.cos(x * x)


x = jnp.array(2.0, dtype=DTYPE)
f_x = f(x)
f_prime_automatic = jax.grad(f)(x)
f_prime_analytical = analytical_derivative(x)

print(f"f({x}) = sin({x}²) = {f_x:.6f}")
print(f"f'({x}) automatic = {f_prime_automatic:.6f}")
print(f"f'({x}) analytical = {f_prime_analytical:.6f}")
print(f"Error: {jnp.abs(f_prime_automatic - f_prime_analytical):.10f}")

viz_stage("stage_1", locals())
print("\n--- Multivariable Gradients ---")


def rosenbrock(params):
    x0, y0 = params[0], params[1]
    a, b = 1.0, 100.0
    return (a - x0) ** 2 + b * (y0 - x0**2) ** 2


grad_rosenbrock = jax.grad(rosenbrock)
point = jnp.array([0.0, 0.0], dtype=DTYPE)
value = rosenbrock(point)
gradient = grad_rosenbrock(point)

print(f"Rosenbrock at {point}: f = {value:.6f}")
print(f"Gradient: ∇f = {gradient}")
print(f"Gradient magnitude: ||∇f|| = {jnp.sqrt(jnp.sum(gradient**2)):.6f}")

viz_stage("stage_final", locals())
