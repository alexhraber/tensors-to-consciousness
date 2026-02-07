# 3_neural_theory.py (JAX)
import jax
import jax.numpy as jnp

from utils import DTYPE, gelu, init_linear, linear, normal, relu, sigmoid, tree_l2_norm, uniform, viz_stage

VIZ_META = {}

print("🧠 NEURAL NETWORK THEORY (JAX)")
print("=" * 50)

print("\n--- Universal Approximation Theory ---")


def target_function(x):
    return jnp.sin(2 * jnp.pi * x) + 0.5 * jnp.cos(4 * jnp.pi * x)


class UniversalApproximator:
    def __init__(self, hidden_size):
        self.params = {
            "l1": init_linear(1, hidden_size),
            "l2": init_linear(hidden_size, hidden_size),
            "l3": init_linear(hidden_size, 1),
        }

    def __call__(self, x, params=None):
        p = self.params if params is None else params
        x = relu(linear(p["l1"], x))
        x = relu(linear(p["l2"], x))
        x = linear(p["l3"], x)
        return x


def analyze_layer_activations(model, x):
    activations = [x]

    current = linear(model.params["l1"], x)
    print(f"Layer 1 (linear): mean={jnp.mean(current):.4f}, std={jnp.std(current):.4f}")
    current = relu(current)
    print(f"Layer 1 (ReLU):   mean={jnp.mean(current):.4f}, std={jnp.std(current):.4f}")
    activations.append(current)

    current = linear(model.params["l2"], current)
    print(f"Layer 2 (linear): mean={jnp.mean(current):.4f}, std={jnp.std(current):.4f}")
    current = relu(current)
    print(f"Layer 2 (ReLU):   mean={jnp.mean(current):.4f}, std={jnp.std(current):.4f}")
    activations.append(current)

    current = linear(model.params["l3"], current)
    print(f"Layer 3 (linear): mean={jnp.mean(current):.4f}, std={jnp.std(current):.4f}")
    activations.append(current)

    return activations


model = UniversalApproximator(32)
test_input = normal((10, 1), dtype=DTYPE)
activations = analyze_layer_activations(model, test_input)

viz_stage("stage_1", locals())
print("\n--- Gradient Flow Analysis ---")


def loss_function(params, x, y):
    predictions = model(x, params=params)
    return jnp.mean((predictions - y) ** 2)


x_train = uniform(-1, 1, (100, 1), dtype=DTYPE)
y_train = target_function(x_train)
gradients = jax.grad(loss_function)(model.params, x_train, y_train)

print("Gradient magnitudes by parameter:")
for layer_name, layer_grads in gradients.items():
    grad_norm = tree_l2_norm(layer_grads)
    print(f"{layer_name}: ||∇|| = {grad_norm:.6f}")

viz_stage("stage_2", locals())
print("\n--- Activation Function Analysis ---")


def compare_activations(x):
    activations = {
        "relu": relu(x),
        "tanh": jnp.tanh(x),
        "sigmoid": sigmoid(x),
        "gelu": gelu(x),
    }

    for name, activation in activations.items():
        mean_val = jnp.mean(activation)
        std_val = jnp.std(activation)
        print(f"{name:>8}: mean={mean_val:.4f}, std={std_val:.4f}")


test_x = normal((1000,), dtype=DTYPE)
compare_activations(test_x)

viz_stage("stage_final", locals())
