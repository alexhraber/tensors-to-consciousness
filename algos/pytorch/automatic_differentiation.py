# 1_automatic_differentiation.py (PyTorch)
import torch

from frameworks.pytorch.utils import DTYPE, scalar, viz_stage

VIZ_META = {}

print("∇ AUTOMATIC DIFFERENTIATION THEORY (PyTorch)")
print("=" * 50)

print("\n--- Chain Rule in Action ---")


def f(x):
    return torch.sin(x * x)


def analytical_derivative(x):
    return 2 * x * torch.cos(x * x)


x = torch.tensor(2.0, dtype=DTYPE, requires_grad=True)
f_x = f(x)
f_x.backward()
f_prime_automatic = x.grad.detach().clone()
f_prime_analytical = analytical_derivative(x.detach())

print(f"f({scalar(x):.1f}) = sin({scalar(x):.1f}²) = {scalar(f_x):.6f}")
print(f"f'({scalar(x):.1f}) automatic = {scalar(f_prime_automatic):.6f}")
print(f"f'({scalar(x):.1f}) analytical = {scalar(f_prime_analytical):.6f}")
print(f"Error: {abs(scalar(f_prime_automatic - f_prime_analytical)):.10f}")

viz_stage("stage_1", locals())
print("\n--- Multivariable Gradients ---")


def rosenbrock(params):
    x0, y0 = params[0], params[1]
    a, b = 1.0, 100.0
    return (a - x0) ** 2 + b * (y0 - x0**2) ** 2


point = torch.tensor([0.0, 0.0], dtype=DTYPE, requires_grad=True)
value = rosenbrock(point)
value.backward()
gradient = point.grad.detach().clone()

print(f"Rosenbrock at {point.detach()}: f = {scalar(value):.6f}")
print(f"Gradient: ∇f = {gradient}")
print(f"Gradient magnitude: ||∇f|| = {scalar(torch.sqrt(torch.sum(gradient**2))):.6f}")

viz_stage("stage_final", locals())
