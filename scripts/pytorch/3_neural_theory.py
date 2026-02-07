# 3_neural_theory.py (PyTorch)
import torch
import torch.nn as nn

from utils import DTYPE, gelu, normal, scalar, sigmoid, tree_l2_norm, uniform, viz_stage

print("🧠 NEURAL NETWORK THEORY (PyTorch)")
print("=" * 50)

print("\n--- Universal Approximation Theory ---")


def target_function(x):
    return torch.sin(2 * torch.pi * x) + 0.5 * torch.cos(4 * torch.pi * x)


class UniversalApproximator(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(1, hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.Linear(hidden_size, 1),
            ]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


def analyze_layer_activations(model, x):
    activations = [x]
    current = x

    for i, layer in enumerate(model.layers):
        current = layer(current)
        print(f"Layer {i + 1} (linear): mean={scalar(torch.mean(current)):.4f}, std={scalar(torch.std(current)):.4f}")
        if i < len(model.layers) - 1:
            current = torch.relu(current)
            print(f"Layer {i + 1} (ReLU):   mean={scalar(torch.mean(current)):.4f}, std={scalar(torch.std(current)):.4f}")
        activations.append(current)

    return activations


model = UniversalApproximator(32).to(dtype=DTYPE)
test_input = normal((10, 1), dtype=DTYPE)
activations = analyze_layer_activations(model, test_input)

viz_stage("stage_1", locals())
print("\n--- Gradient Flow Analysis ---")


def loss_function(model, x, y):
    predictions = model(x)
    return torch.mean((predictions - y) ** 2)


x_train = uniform(-1, 1, (100, 1), dtype=DTYPE)
y_train = target_function(x_train)

for p in model.parameters():
    if p.grad is not None:
        p.grad.zero_()

loss = loss_function(model, x_train, y_train)
loss.backward()

print("Gradient magnitudes by parameter:")
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = tree_l2_norm([param.grad])
        print(f"{name}: ||∇|| = {scalar(grad_norm):.6f}")

viz_stage("stage_2", locals())
print("\n--- Activation Function Analysis ---")


def compare_activations(x):
    activations = {
        "relu": torch.relu(x),
        "tanh": torch.tanh(x),
        "sigmoid": sigmoid(x),
        "gelu": gelu(x),
    }

    for name, activation in activations.items():
        mean_val = torch.mean(activation)
        std_val = torch.std(activation)
        print(f"{name:>8}: mean={scalar(mean_val):.4f}, std={scalar(std_val):.4f}")


test_x = normal((1000,), dtype=DTYPE)
compare_activations(test_x)

viz_stage("stage_final", locals())
