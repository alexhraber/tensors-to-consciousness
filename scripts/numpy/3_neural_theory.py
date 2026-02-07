# 3_neural_theory.py (NumPy)
import numpy as np

from utils import DTYPE, finite_diff_grad_dict, gelu, init_linear, linear, normal, relu, sigmoid, tree_l2_norm, uniform, viz_stage

VIZ_META = {}

print("🧠 NEURAL NETWORK THEORY (NumPy)")
print("=" * 50)

print("\n--- Universal Approximation Theory ---")


def target_function(x):
    return np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x)


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
    print(f"Layer 1 (linear): mean={np.mean(current):.4f}, std={np.std(current):.4f}")
    current = relu(current)
    print(f"Layer 1 (ReLU):   mean={np.mean(current):.4f}, std={np.std(current):.4f}")
    activations.append(current)

    current = linear(model.params["l2"], current)
    print(f"Layer 2 (linear): mean={np.mean(current):.4f}, std={np.std(current):.4f}")
    current = relu(current)
    print(f"Layer 2 (ReLU):   mean={np.mean(current):.4f}, std={np.std(current):.4f}")
    activations.append(current)

    current = linear(model.params["l3"], current)
    print(f"Layer 3 (linear): mean={np.mean(current):.4f}, std={np.std(current):.4f}")
    activations.append(current)

    return activations


model = UniversalApproximator(16)
test_input = normal((10, 1), dtype=DTYPE)
activations = analyze_layer_activations(model, test_input)

viz_stage("stage_1", locals())
print("\n--- Gradient Flow Analysis ---")


def flatten_params(params):
    return {
        "l1_weight": params["l1"]["weight"],
        "l1_bias": params["l1"]["bias"],
        "l2_weight": params["l2"]["weight"],
        "l2_bias": params["l2"]["bias"],
        "l3_weight": params["l3"]["weight"],
        "l3_bias": params["l3"]["bias"],
    }


def unflatten_params(flat):
    return {
        "l1": {"weight": flat["l1_weight"], "bias": flat["l1_bias"]},
        "l2": {"weight": flat["l2_weight"], "bias": flat["l2_bias"]},
        "l3": {"weight": flat["l3_weight"], "bias": flat["l3_bias"]},
    }


def loss_function(flat_params, x, y):
    params = unflatten_params(flat_params)
    predictions = model(x, params=params)
    return np.mean((predictions - y) ** 2)


x_train = uniform(-1, 1, (64, 1), dtype=DTYPE)
y_train = target_function(x_train)
flat = flatten_params(model.params)


def wrapped_loss(p):
    return loss_function(p, x_train, y_train)


grads = finite_diff_grad_dict(wrapped_loss, flat)

print("Gradient magnitudes by parameter:")
for name, grad in grads.items():
    print(f"{name}: ||∇|| = {tree_l2_norm([grad]):.6f}")

viz_stage("stage_2", locals())
print("\n--- Activation Function Analysis ---")


def compare_activations(x):
    activations = {
        "relu": relu(x),
        "tanh": np.tanh(x),
        "sigmoid": sigmoid(x),
        "gelu": gelu(x),
    }

    for name, activation in activations.items():
        mean_val = np.mean(activation)
        std_val = np.std(activation)
        print(f"{name:>8}: mean={mean_val:.4f}, std={std_val:.4f}")


test_x = normal((1000,), dtype=DTYPE)
compare_activations(test_x)

viz_stage("stage_final", locals())
