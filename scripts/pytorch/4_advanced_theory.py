# 4_advanced_theory.py (PyTorch)
import torch
import torch.nn as nn

from utils import DTYPE, normal, scalar, softmax, uniform, viz_stage

VIZ_META = {}

print("🔬 ADVANCED COMPUTATIONAL THEORY (PyTorch)")
print("=" * 50)

print("\n--- Manifold Learning Theory ---")


def generate_swiss_roll(n_samples):
    t = uniform(0, 4 * torch.pi, (n_samples,), dtype=DTYPE)
    height = uniform(-10, 10, (n_samples,), dtype=DTYPE)

    x = t * torch.cos(t)
    y = height
    z = t * torch.sin(t)

    return torch.stack([x, y, z], dim=1)


swiss_data = generate_swiss_roll(1000)
print(f"Swiss roll shape: {swiss_data.shape}")
print("Intrinsic dimension: 2, Ambient dimension: 3")
print(f"Data range: [{scalar(torch.min(swiss_data)):.2f}, {scalar(torch.max(swiss_data)):.2f}]")

viz_stage("stage_1", locals())
print("\n--- Information Bottleneck Theory ---")


class InformationBottleneck(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, output_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, bottleneck_dim)
        self.decoder = nn.Linear(bottleneck_dim, output_dim)
        self.bottleneck_dim = bottleneck_dim

    def forward(self, x):
        z = torch.tanh(self.encoder(x))
        output = self.decoder(z)
        return output, z


models = {
    "high_capacity": InformationBottleneck(10, 8, 5),
    "medium_capacity": InformationBottleneck(10, 4, 5),
    "low_capacity": InformationBottleneck(10, 2, 5),
}

for model in models.values():
    model.to(dtype=DTYPE)

test_data = normal((100, 10), dtype=DTYPE)
for name, model in models.items():
    output, bottleneck = model(test_data)
    compression_ratio = 10 / model.bottleneck_dim
    bottleneck_info = torch.std(bottleneck)
    print(f"{name}: compression {compression_ratio:.1f}x, info={scalar(bottleneck_info):.4f}")

viz_stage("stage_2", locals())
print("\n--- Attention Theory ---")


def attention_mechanism(queries, keys, values, temperature=1.0):
    scores = torch.matmul(queries, keys.T) / torch.sqrt(torch.tensor(queries.shape[-1], dtype=DTYPE))
    scores = scores / temperature
    attention_weights = softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, values)
    return output, attention_weights


queries = normal((3, 64), dtype=DTYPE)
keys = normal((4, 64), dtype=DTYPE)
values = normal((4, 64), dtype=DTYPE)

attended_output, weights = attention_mechanism(queries, keys, values)
print(f"Attention weights shape: {weights.shape}")
print(f"Attention weights (row sums): {torch.sum(weights, dim=1)}")
print(f"Max attention per query: {torch.max(weights, dim=1).values}")

viz_stage("stage_3", locals())
print("\n--- Riemannian Optimization ---")


def project_to_sphere(x):
    return x / (torch.sqrt(torch.sum(x**2)) + 1e-8)


def riemannian_gradient_step(x, grad, lr=0.01):
    tangent_grad = grad - torch.sum(grad * x) * x
    x_new = x - lr * tangent_grad
    return project_to_sphere(x_new)


def sphere_objective(x):
    target = torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE)
    return torch.sum((x - target) ** 2)


x = project_to_sphere(normal((3,), dtype=DTYPE))
print("Optimizing on sphere manifold:")
print(f"Initial: obj={scalar(sphere_objective(x)):.6f}, ||x||={scalar(torch.sqrt(torch.sum(x**2))):.6f}")

for i in range(10):
    x_var = x.clone().detach().requires_grad_(True)
    obj = sphere_objective(x_var)
    obj.backward()
    grad = x_var.grad.detach()
    x = riemannian_gradient_step(x_var.detach(), grad)
    if i % 3 == 0:
        obj_val = sphere_objective(x)
        constraint_violation = torch.abs(torch.sum(x**2) - 1.0)
        print(f"Step {i + 1}: obj={scalar(obj_val):.6f}, constraint_error={scalar(constraint_violation):.8f}")

print(f"Final point: {x}")

viz_stage("stage_final", locals())
