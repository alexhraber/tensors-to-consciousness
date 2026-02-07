# 5_research_frontiers.py (PyTorch)
import torch
import torch.nn as nn

from utils import DTYPE, normal, scalar

print("🚀 RESEARCH FRONTIERS (PyTorch)")
print("=" * 50)

print("\n--- Meta-Learning Theory ---")


class SimpleMetaLearner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.base_layer = nn.Linear(dim, dim)
        self.meta_layer = nn.Linear(dim, dim)

    def forward(self, x):
        return self.base_layer(x)

    def meta_forward(self, x, adaptation_signal):
        adaptation = self.meta_layer(adaptation_signal)
        base_output = self.base_layer(x)
        return base_output + adaptation


meta_model = SimpleMetaLearner(10).to(dtype=DTYPE)
x = normal((5, 10), dtype=DTYPE)
adaptation_signal = normal((5, 10), dtype=DTYPE)

base_output = meta_model(x)
adapted_output = meta_model.meta_forward(x, adaptation_signal)

print(f"Base output norm: {scalar(torch.sqrt(torch.sum(base_output**2))):.4f}")
print(f"Adapted output norm: {scalar(torch.sqrt(torch.sum(adapted_output**2))):.4f}")
print(f"Adaptation magnitude: {scalar(torch.sqrt(torch.sum((adapted_output - base_output) ** 2))):.4f}")

print("\n--- Neural Scaling Laws ---")


def estimate_scaling_law(model_sizes, data_sizes):
    results = {}
    for model_size in model_sizes:
        for data_size in data_sizes:
            alpha, beta = 0.076, 0.095
            A, B, E = 1.0, 1.0, 1.3
            loss = A * (model_size ** -alpha) + B * (data_size ** -beta) + E
            results[(model_size, data_size)] = loss
    return results


model_sizes = [1e6, 1e7, 1e8, 1e9]
data_sizes = [1e6, 1e7, 1e8, 1e9]
scaling_results = estimate_scaling_law(model_sizes, data_sizes)

print("Scaling law predictions (loss vs model/data size):")
for (model_size, data_size), loss in list(scaling_results.items())[:4]:
    print(f"Model: {model_size:.0e}, Data: {data_size:.0e} → Loss: {loss:.4f}")

print("\n--- Lottery Ticket Hypothesis ---")


def find_lottery_ticket(model, pruning_ratio=0.9):
    all_weights = []

    for name, param in model.named_parameters():
        if "weight" in name:
            all_weights.append(param.detach().flatten())

    if not all_weights:
        print("No weights found to prune")
        return None

    weight_vector = torch.cat(all_weights)
    sorted_weights = torch.sort(torch.abs(weight_vector)).values
    threshold_idx = int(len(sorted_weights) * pruning_ratio)
    threshold = sorted_weights[threshold_idx]

    mask = torch.abs(weight_vector) > threshold
    sparsity = 1.0 - torch.mean(mask.to(dtype=DTYPE))

    print(f"Lottery ticket sparsity: {scalar(sparsity):.1%}")
    print(f"Remaining parameters: {int(torch.sum(mask))} / {len(weight_vector)}")
    return mask


test_model = nn.Linear(100, 50).to(dtype=DTYPE)
lottery_mask = find_lottery_ticket(test_model)

print("\n--- Grokking Phenomenon ---")


def simulate_grokking_dynamics(steps=1000):
    memorization_phase = 300
    generalization_phase = 700

    train_losses = []
    test_losses = []

    for step in range(steps):
        if step < memorization_phase:
            train_loss = 2.0 * torch.exp(torch.tensor(-step / 100, dtype=DTYPE))
            test_loss = torch.tensor(2.0, dtype=DTYPE) + 0.1 * normal((), dtype=DTYPE)
        elif step < generalization_phase:
            prev_train = train_losses[-1] if train_losses else 0.5
            prev_test = test_losses[-1] if test_losses else 2.0
            train_loss = torch.tensor(prev_train, dtype=DTYPE) + 0.1 * normal((), dtype=DTYPE)
            test_loss = torch.tensor(prev_test, dtype=DTYPE) + 0.2 * normal((), dtype=DTYPE)
        else:
            progress = (step - generalization_phase) / (steps - generalization_phase)
            train_loss = torch.tensor(0.1, dtype=DTYPE) + 0.05 * torch.exp(torch.tensor(-progress * 5, dtype=DTYPE))
            test_loss = torch.tensor(0.1, dtype=DTYPE) + 0.05 * torch.exp(torch.tensor(-progress * 5, dtype=DTYPE))

        train_losses.append(scalar(train_loss))
        test_losses.append(scalar(test_loss))

    return train_losses, test_losses


train_losses, test_losses = simulate_grokking_dynamics()
print("Grokking simulation:")
print(f"Pre-grokking test loss: {test_losses[300]:.4f}")
print(f"Post-grokking test loss: {test_losses[-1]:.4f}")
print(f"Generalization improvement: {test_losses[300] / test_losses[-1]:.1f}x")

print("\n--- Emergent Capabilities ---")


def model_capability_curve(model_sizes, task_complexity):
    capabilities = []
    for size in model_sizes:
        threshold = task_complexity * 1e8
        steepness = 10
        exponent = -steepness * (size - threshold) / threshold
        capability = 1 / (1 + torch.exp(torch.tensor(exponent, dtype=DTYPE)))
        capabilities.append(scalar(capability))
    return capabilities


model_sizes_array = [1e6, 1e7, 1e8, 1e9, 1e10]
tasks = {
    "simple_arithmetic": 0.1,
    "complex_reasoning": 1.0,
    "few_shot_learning": 2.0,
    "emergent_reasoning": 5.0,
}

print("Emergent capabilities by model size:")
for task_name, complexity in tasks.items():
    capabilities = model_capability_curve(model_sizes_array, complexity)
    for i, (size, capability) in enumerate(zip(model_sizes_array, capabilities)):
        if i % 2 == 0:
            print(f"{task_name} @ {size:.0e} params: {capability:.3f}")
