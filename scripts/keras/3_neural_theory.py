# 3_neural_theory.py (Keras)
import numpy as np
import tensorflow as tf
import keras
from keras import ops

from utils import DTYPE, gelu, normal, relu, scalar, sigmoid, uniform, viz_stage

print("🧠 NEURAL NETWORK THEORY (Keras)")
print("=" * 50)

print("\n--- Universal Approximation Theory ---")


def target_function(x):
    return ops.sin(2 * np.pi * x) + 0.5 * ops.cos(4 * np.pi * x)


class UniversalApproximator(keras.Model):
    def __init__(self, hidden_size):
        super().__init__()
        self.layers_list = [
            keras.layers.Dense(hidden_size, dtype=DTYPE),
            keras.layers.Dense(hidden_size, dtype=DTYPE),
            keras.layers.Dense(1, dtype=DTYPE),
        ]

    def call(self, x):
        for i, layer in enumerate(self.layers_list):
            x = layer(x)
            if i < len(self.layers_list) - 1:
                x = relu(x)
        return x


def analyze_layer_activations(model, x):
    activations = [x]
    current = x

    for i, layer in enumerate(model.layers_list):
        current = layer(current)
        print(f"Layer {i + 1} (linear): mean={scalar(ops.mean(current)):.4f}, std={scalar(ops.std(current)):.4f}")
        if i < len(model.layers_list) - 1:
            current = relu(current)
            print(f"Layer {i + 1} (ReLU):   mean={scalar(ops.mean(current)):.4f}, std={scalar(ops.std(current)):.4f}")
        activations.append(current)

    return activations


model = UniversalApproximator(32)
test_input = normal((10, 1), dtype=DTYPE)
_ = model(test_input)
activations = analyze_layer_activations(model, test_input)

viz_stage("stage_1", locals())
print("\n--- Gradient Flow Analysis ---")


def loss_function(model, x, y):
    predictions = model(x)
    return ops.mean((predictions - y) ** 2)


x_train = uniform(-1, 1, (100, 1), dtype=DTYPE)
y_train = target_function(x_train)

with tf.GradientTape() as tape:
    loss = loss_function(model, x_train, y_train)
gradients = tape.gradient(loss, model.trainable_variables)

print("Gradient magnitudes by parameter:")
for var, grad in zip(model.trainable_variables, gradients):
    if grad is not None:
        grad_norm = ops.sqrt(ops.sum(grad * grad))
        print(f"{var.name}: ||∇|| = {scalar(grad_norm):.6f}")

viz_stage("stage_2", locals())
print("\n--- Activation Function Analysis ---")


def compare_activations(x):
    activations = {
        "relu": relu(x),
        "tanh": ops.tanh(x),
        "sigmoid": sigmoid(x),
        "gelu": gelu(x),
    }

    for name, activation in activations.items():
        mean_val = ops.mean(activation)
        std_val = ops.std(activation)
        print(f"{name:>8}: mean={scalar(mean_val):.4f}, std={scalar(std_val):.4f}")


test_x = normal((1000,), dtype=DTYPE)
compare_activations(test_x)

viz_stage("stage_final", locals())
