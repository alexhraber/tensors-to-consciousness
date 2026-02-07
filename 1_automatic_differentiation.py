# 1_automatic_differentiation.py
import t2c.frameworks as fw

mx = fw.mx

print("∇ AUTOMATIC DIFFERENTIATION THEORY")
print("=" * 50)

# Set global dtype for consistency
DTYPE = mx.float32

# 1. THE CHAIN RULE: Foundation of backpropagation
print("\n--- Chain Rule in Action ---")

def f(x):
    """f(x) = sin(x²)"""
    return mx.sin(x * x)

def analytical_derivative(x):
    """f'(x) = 2x * cos(x²) - analytical solution"""
    return 2 * x * mx.cos(x * x)

# Automatic differentiation vs analytical
x = mx.array(2.0, dtype=DTYPE)
f_x = f(x)
f_prime_automatic = mx.grad(f)(x)
f_prime_analytical = analytical_derivative(x)

print(f"f({x}) = sin({x}²) = {f_x:.6f}")
print(f"f'({x}) automatic = {f_prime_automatic:.6f}")
print(f"f'({x}) analytical = {f_prime_analytical:.6f}")
print(f"Error: {abs(f_prime_automatic - f_prime_analytical):.10f}")

# 2. MULTIVARIABLE GRADIENTS: The foundation of optimization
print("\n--- Multivariable Gradients ---")

def rosenbrock(params):
    """Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²"""
    x, y = params[0], params[1]
    a, b = 1.0, 100.0
    return (a - x)**2 + b * (y - x**2)**2

# Gradient gives direction of steepest ascent
grad_rosenbrock = mx.grad(rosenbrock)

point = mx.array([0.0, 0.0], dtype=DTYPE)
value = rosenbrock(point)
gradient = grad_rosenbrock(point)

print(f"Rosenbrock at {point}: f = {value:.6f}")
print(f"Gradient: ∇f = {gradient}")
print(f"Gradient magnitude: ||∇f|| = {mx.sqrt(mx.sum(gradient**2)):.6f}")
