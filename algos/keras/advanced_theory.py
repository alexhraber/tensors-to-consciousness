# 4_advanced_theory.py (Keras)
import numpy as np
import keras
from keras import ops

from frameworks.keras.utils import DTYPE, finite_diff_grad_vector, normal, scalar, softmax, uniform, viz_stage

VIZ_META = {}

print("🔬 ADVANCED COMPUTATIONAL THEORY (Keras)")
print("=" * 50)

print("\n--- Manifold Learning Theory ---")


def generate_swiss_roll(n_samples):
    t = uniform(0, 4 * np.pi, (n_samples,), dtype=DTYPE)
    height = uniform(-10, 10, (n_samples,), dtype=DTYPE)

    x = t * ops.cos(t)
    y = height
    z = t * ops.sin(t)

    return ops.stack([x, y, z], axis=1)


swiss_data = generate_swiss_roll(1000)
print(f"Swiss roll shape: {swiss_data.shape}")
print("Intrinsic dimension: 2, Ambient dimension: 3")
print(f"Data range: [{scalar(ops.min(swiss_data)):.2f}, {scalar(ops.max(swiss_data)):.2f}]")

viz_stage("stage_1", locals())
print("\n--- Information Bottleneck Theory ---")


class InformationBottleneck(keras.Model):
    def __init__(self, input_dim, bottleneck_dim, output_dim):
        super().__init__()
        self.encoder = keras.layers.Dense(bottleneck_dim, dtype=DTYPE)
        self.decoder = keras.layers.Dense(output_dim, dtype=DTYPE)
        self.bottleneck_dim = bottleneck_dim

    def call(self, x):
        z = ops.tanh(self.encoder(x))
        output = self.decoder(z)
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
    bottleneck_info = ops.std(bottleneck)
    print(f"{name}: compression {compression_ratio:.1f}x, info={scalar(bottleneck_info):.4f}")

viz_stage("stage_2", locals())
print("\n--- Attention Theory ---")


def attention_mechanism(queries, keys, values, temperature=1.0):
    scores = ops.matmul(queries, ops.transpose(keys)) / ops.sqrt(ops.array(queries.shape[-1], dtype=DTYPE))
    scores = scores / temperature
    attention_weights = softmax(scores, axis=-1)
    output = ops.matmul(attention_weights, values)
    return output, attention_weights


queries = normal((3, 64), dtype=DTYPE)
keys = normal((4, 64), dtype=DTYPE)
values = normal((4, 64), dtype=DTYPE)

attended_output, weights = attention_mechanism(queries, keys, values)
print(f"Attention weights shape: {weights.shape}")
print(f"Attention weights (row sums): {ops.sum(weights, axis=1)}")
print(f"Max attention per query: {ops.max(weights, axis=1)}")

viz_stage("stage_3", locals())
print("\n--- Riemannian Optimization ---")


def project_to_sphere(x):
    return x / (np.sqrt(np.sum(x**2)) + 1e-8)


def riemannian_gradient_step(x, grad, lr=0.01):
    tangent_grad = grad - np.sum(grad * x) * x
    x_new = x - lr * tangent_grad
    return project_to_sphere(x_new)


def sphere_objective(x):
    target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return np.sum((x - target) ** 2)


x = project_to_sphere(np.array(ops.convert_to_numpy(normal((3,), dtype=DTYPE))))
print("Optimizing on sphere manifold:")
print(f"Initial: obj={sphere_objective(x):.6f}, ||x||={np.sqrt(np.sum(x**2)):.6f}")

for i in range(10):
    grad = finite_diff_grad_vector(sphere_objective, x)
    x = riemannian_gradient_step(x, grad)
    if i % 3 == 0:
        obj_val = sphere_objective(x)
        constraint_violation = abs(np.sum(x**2) - 1.0)
        print(f"Step {i + 1}: obj={obj_val:.6f}, constraint_error={constraint_violation:.8f}")

print(f"Final point: {x}")

viz_stage("stage_final", locals())
