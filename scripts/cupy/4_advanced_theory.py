# 4_advanced_theory.py (CuPy)
import cupy as cp

from utils import DTYPE, finite_diff_grad_vector, init_linear, linear, normal, softmax, uniform, viz_stage

VIZ_META = {}

print("🔬 ADVANCED COMPUTATIONAL THEORY (CuPy)")
print("=" * 50)

print("\n--- Manifold Learning Theory ---")


def generate_swiss_roll(n_samples):
    t = uniform(0, 4 * cp.pi, (n_samples,), dtype=DTYPE)
    height = uniform(-10, 10, (n_samples,), dtype=DTYPE)

    x = t * cp.cos(t)
    y = height
    z = t * cp.sin(t)

    return cp.stack([x, y, z], axis=1)


swiss_data = generate_swiss_roll(1000)
print(f"Swiss roll shape: {swiss_data.shape}")
print("Intrinsic dimension: 2, Ambient dimension: 3")
print(f"Data range: [{cp.min(swiss_data):.2f}, {cp.max(swiss_data):.2f}]")

viz_stage("stage_1", locals())
print("\n--- Information Bottleneck Theory ---")


class InformationBottleneck:
    def __init__(self, input_dim, bottleneck_dim, output_dim):
        self.encoder = init_linear(input_dim, bottleneck_dim)
        self.decoder = init_linear(bottleneck_dim, output_dim)
        self.bottleneck_dim = bottleneck_dim

    def __call__(self, x):
        z = cp.tanh(linear(self.encoder, x))
        output = linear(self.decoder, z)
        return output, z


models = {
    "high_capacity": InformationBottleneck(10, 8, 5),
    "medium_capacity": InformationBottleneck(10, 4, 5),
    "low_capacity": InformationBottleneck(10, 2, 5),
}

test_data = normal((100, 10), dtype=DTYPE)
for name, model in models.items():
    output, bottleneck = model(test_data)
    compression_ratio = 10 / model.bottleneck_dim
    bottleneck_info = cp.std(bottleneck)
    print(f"{name}: compression {compression_ratio:.1f}x, info={bottleneck_info:.4f}")

viz_stage("stage_2", locals())
print("\n--- Attention Theory ---")


def attention_mechanism(queries, keys, values, temperature=1.0):
    scores = cp.matmul(queries, keys.T) / cp.sqrt(cp.array(queries.shape[-1], dtype=DTYPE))
    scores = scores / temperature
    attention_weights = softmax(scores, axis=-1)
    output = cp.matmul(attention_weights, values)
    return output, attention_weights


queries = normal((3, 64), dtype=DTYPE)
keys = normal((4, 64), dtype=DTYPE)
values = normal((4, 64), dtype=DTYPE)

attended_output, weights = attention_mechanism(queries, keys, values)
print(f"Attention weights shape: {weights.shape}")
print(f"Attention weights (row sums): {cp.sum(weights, axis=1)}")
print(f"Max attention per query: {cp.max(weights, axis=1)}")

viz_stage("stage_3", locals())
print("\n--- Riemannian Optimization ---")


def project_to_sphere(x):
    return x / (cp.sqrt(cp.sum(x**2)) + 1e-8)


def riemannian_gradient_step(x, grad, lr=0.01):
    tangent_grad = grad - cp.sum(grad * x) * x
    x_new = x - lr * tangent_grad
    return project_to_sphere(x_new)


def sphere_objective(x):
    target = cp.array([1.0, 0.0, 0.0], dtype=DTYPE)
    return cp.sum((x - target) ** 2)


x = project_to_sphere(normal((3,), dtype=DTYPE))
print("Optimizing on sphere manifold:")
print(f"Initial: obj={sphere_objective(x):.6f}, ||x||={cp.sqrt(cp.sum(x**2)):.6f}")

for i in range(10):
    grad = finite_diff_grad_vector(sphere_objective, x)
    x = riemannian_gradient_step(x, grad)
    if i % 3 == 0:
        obj_val = sphere_objective(x)
        constraint_violation = cp.abs(cp.sum(x**2) - 1.0)
        print(f"Step {i + 1}: obj={obj_val:.6f}, constraint_error={constraint_violation:.8f}")

print(f"Final point: {x}")

viz_stage("stage_final", locals())
