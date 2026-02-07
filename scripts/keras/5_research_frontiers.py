# 5_research_frontiers.py (Keras)
import numpy as np
import keras
from keras import ops

from utils import DTYPE, normal, scalar

print("🚀 RESEARCH FRONTIERS (Keras)")
print("=" * 50)

print("\n--- Meta-Learning Theory ---")


class SimpleMetaLearner(keras.Model):
    def __init__(self, dim):
        super().__init__()
        self.base_layer = keras.layers.Dense(dim, dtype=DTYPE)
        self.meta_layer = keras.layers.Dense(dim, dtype=DTYPE)

    def call(self, x):
        return self.base_layer(x)

    def meta_forward(self, x, adaptation_signal):
        adaptation = self.meta_layer(adaptation_signal)
        base_output = self.base_layer(x)
        return base_output + adaptation


meta_model = SimpleMetaLearner(10)
x = normal((5, 10), dtype=DTYPE)
adaptation_signal = normal((5, 10), dtype=DTYPE)
_ = meta_model(x)

base_output = meta_model(x)
adapted_output = meta_model.meta_forward(x, adaptation_signal)

print(f"Base output norm: {scalar(ops.sqrt(ops.sum(base_output**2))):.4f}")
print(f"Adapted output norm: {scalar(ops.sqrt(ops.sum(adapted_output**2))):.4f}")
print(f"Adaptation magnitude: {scalar(ops.sqrt(ops.sum((adapted_output - base_output) ** 2))):.4f}")

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
    for var in model.trainable_variables:
        if "kernel" in var.name:
            all_weights.append(ops.reshape(var, (-1,)))

    if not all_weights:
        print("No weights found to prune")
        return None

    weight_vector = ops.concatenate(all_weights, axis=0)
    sorted_weights = ops.sort(ops.abs(weight_vector))
    threshold_idx = int(sorted_weights.shape[0] * pruning_ratio)
    threshold = sorted_weights[threshold_idx]

    mask = ops.abs(weight_vector) > threshold
    sparsity = 1.0 - scalar(ops.mean(ops.cast(mask, DTYPE)))

    print(f"Lottery ticket sparsity: {sparsity:.1%}")
    print(f"Remaining parameters: {scalar(ops.sum(ops.cast(mask, DTYPE))):.0f} / {sorted_weights.shape[0]}")
    return mask


test_model = keras.layers.Dense(50, dtype=DTYPE)
_ = test_model(normal((1, 100), dtype=DTYPE))
lottery_mask = find_lottery_ticket(test_model)

print("\n--- Grokking Phenomenon ---")


def simulate_grokking_dynamics(steps=1000):
    memorization_phase = 300
    generalization_phase = 700

    train_losses = []
    test_losses = []

    for step in range(steps):
        if step < memorization_phase:
            train_loss = 2.0 * np.exp(-step / 100)
            test_loss = 2.0 + 0.1 * np.random.randn()
        elif step < generalization_phase:
            prev_train = train_losses[-1] if train_losses else 0.5
            prev_test = test_losses[-1] if test_losses else 2.0
            train_loss = prev_train + 0.1 * np.random.randn()
            test_loss = prev_test + 0.2 * np.random.randn()
        else:
            progress = (step - generalization_phase) / (steps - generalization_phase)
            train_loss = 0.1 + 0.05 * np.exp(-progress * 5)
            test_loss = 0.1 + 0.05 * np.exp(-progress * 5)

        train_losses.append(float(train_loss))
        test_losses.append(float(test_loss))

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
        capability = 1 / (1 + np.exp(exponent))
        capabilities.append(float(capability))
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
