# 6_theoretical_limits.py (Keras)
import numpy as np
import tensorflow as tf
import keras
from keras import ops

from utils import DTYPE, normal, scalar, softmax, viz_stage

print("🌌 THEORETICAL LIMITS & FUNDAMENTAL BOUNDARIES (Keras)")
print("=" * 50)

print("\n--- Information Geometry & Natural Gradients ---")


def kl_divergence(p, q):
    eps = 1e-8
    p_safe = ops.maximum(p, eps)
    q_safe = ops.maximum(q, eps)
    return ops.sum(p_safe * ops.log(p_safe / q_safe))


simple_model = keras.layers.Dense(3, dtype=DTYPE)
test_data = normal((10, 5), dtype=DTYPE)
_ = simple_model(test_data)

with tf.GradientTape() as tape:
    output = simple_model(test_data)
    log_like = -ops.sum(output**2) / 2

grads = tape.gradient(log_like, simple_model.trainable_variables)
grad_flat = [ops.reshape(g, (-1,)) for g in grads if g is not None]
grad_vector = ops.concatenate(grad_flat, axis=0)
fisher_matrix = ops.matmul(ops.expand_dims(grad_vector, 1), ops.expand_dims(grad_vector, 0))

print(f"Fisher matrix shape: {fisher_matrix.shape}")
fisher_max = ops.max(fisher_matrix)
fisher_min = ops.min(fisher_matrix)
condition_number = fisher_max / (ops.abs(fisher_min) + 1e-8)
print(f"Fisher matrix condition number: {scalar(condition_number):.2f}")

viz_stage("stage_1", locals())
print("\n--- Algorithmic Information Theory ---")


def estimate_kolmogorov_complexity(data):
    complexities = {}
    data_np = np.array(ops.convert_to_numpy(data)).flatten()
    threshold = np.mean(data_np)
    data_binary = (data_np > threshold).astype(np.int32)

    def rle_compress(binary_data):
        runs = []
        if len(binary_data) > 0:
            current = binary_data[0]
            count = 1
            for i in range(1, len(binary_data)):
                if binary_data[i] == current:
                    count += 1
                else:
                    runs.append(count)
                    current = binary_data[i]
                    count = 1
            runs.append(count)
        return len(runs)

    def entropy_compress(binary_data):
        total = len(binary_data)
        ones = np.sum(binary_data)
        zeros = total - ones
        if ones > 0 and zeros > 0:
            p_ones = ones / total
            p_zeros = zeros / total
            entropy = -(p_ones * np.log2(p_ones) + p_zeros * np.log2(p_zeros))
        else:
            entropy = 0.0
        return float(entropy * total / 8)

    complexities["rle"] = rle_compress(data_binary)
    complexities["entropy"] = entropy_compress(data_binary)
    return complexities


random_data = normal((100,), dtype=DTYPE)
structured_data = ops.sin(ops.arange(0, 4 * np.pi, 4 * np.pi / 100, dtype=DTYPE))

random_complexity = estimate_kolmogorov_complexity(random_data)
structured_complexity = estimate_kolmogorov_complexity(structured_data)

print(f"Random data complexity: {random_complexity}")
print(f"Structured data complexity: {structured_complexity}")
entropy_ratio = random_complexity["entropy"] / (structured_complexity["entropy"] + 1e-8)
print(f"Compression ratio (random vs structured): {entropy_ratio:.2f}")

viz_stage("stage_2", locals())
print("\n--- Quantum-Inspired Computation ---")


def quantum_superposition_layer(x, n_qubits=4):
    batch_size, input_dim = x.shape
    real_amplitudes = normal((batch_size, n_qubits), dtype=DTYPE)
    imag_amplitudes = normal((batch_size, n_qubits), dtype=DTYPE)

    amplitude_norms = ops.sqrt(real_amplitudes**2 + imag_amplitudes**2)
    total_norm = ops.sum(amplitude_norms, axis=1, keepdims=True)

    real_normalized = real_amplitudes / (total_norm + 1e-8)
    imag_normalized = imag_amplitudes / (total_norm + 1e-8)

    probabilities = real_normalized**2 + imag_normalized**2
    qubit_indices = ops.reshape(ops.arange(n_qubits, dtype=DTYPE), (1, -1))
    classical_output = ops.sum(probabilities * qubit_indices, axis=1)

    return classical_output, probabilities


quantum_input = normal((5, 8), dtype=DTYPE)
quantum_output, quantum_probs = quantum_superposition_layer(quantum_input)

print(f"Quantum output shape: {quantum_output.shape}")
print(f"Probability conservation check: {ops.sum(quantum_probs, axis=1)}")
print(f"Quantum coherence measure: {scalar(ops.std(quantum_probs)):.4f}")

viz_stage("stage_3", locals())
print("\n--- Integrated Information Theory (Φ) ---")


def compute_phi(network_state, connectivity_matrix):
    ns = np.array(ops.convert_to_numpy(network_state))
    cm = np.array(ops.convert_to_numpy(connectivity_matrix))

    system_entropy = -np.sum(ns * np.log2(ns + 1e-8))

    part_entropies = 0.0
    for node_prob in ns:
        if node_prob > 0 and node_prob < 1:
            part_entropies += -(node_prob * np.log2(node_prob) + (1 - node_prob) * np.log2(1 - node_prob))

    phi = system_entropy - part_entropies
    connectivity_strength = np.sum(np.abs(cm))
    return float(phi), float(connectivity_strength)


connected_state = softmax(normal((6,), dtype=DTYPE), axis=0)
connected_matrix = normal((6, 6), dtype=DTYPE) * 0.8

disconnected_state = softmax(normal((6,), dtype=DTYPE), axis=0)
disconnected_matrix = ops.eye(6, dtype=DTYPE) * 0.1

phi_connected, conn_connected = compute_phi(connected_state, connected_matrix)
phi_disconnected, conn_disconnected = compute_phi(disconnected_state, disconnected_matrix)

print(f"Connected network Φ: {phi_connected:.4f}, connectivity: {conn_connected:.2f}")
print(f"Disconnected network Φ: {phi_disconnected:.4f}, connectivity: {conn_disconnected:.2f}")
consciousness_ratio = phi_connected / (phi_disconnected + 1e-8)
print(f"Consciousness ratio: {consciousness_ratio:.2f}")

viz_stage("stage_4", locals())
print("\n--- Universal Approximation at Infinite Width ---")


def infinite_width_approximation(target_func, x_points, width_schedule):
    approximation_errors = []

    for width in width_schedule:
        model = keras.Sequential(
            [
                keras.layers.Dense(width, activation="tanh", dtype=DTYPE),
                keras.layers.Dense(1, dtype=DTYPE),
            ]
        )
        network_output = model(ops.reshape(x_points, (-1, 1)))
        target_output = ops.reshape(target_func(x_points), (-1, 1))
        error = ops.mean((network_output - target_output) ** 2)
        approximation_errors.append(scalar(error))

    return approximation_errors


def complex_target(x):
    return ops.sin(3 * x) * ops.exp(-(x**2)) + ops.cos(5 * x) * x


x_test = ops.arange(-2, 2, 4 / 100, dtype=DTYPE)
widths = [10, 50, 100, 500, 1000]
errors = infinite_width_approximation(complex_target, x_test, widths)

print("Universal approximation convergence:")
for width, error in zip(widths, errors):
    print(f"Width {width:4d}: Error = {error:.6f}")

viz_stage("stage_5", locals())
print("\n--- Thermodynamics of Learning ---")


thermo_model = keras.layers.Dense(5, dtype=DTYPE)
thermo_data = normal((20, 10), dtype=DTYPE)
_ = thermo_model(thermo_data)

with tf.GradientTape() as tape:
    output = thermo_model(thermo_data)
    loss = ops.sum(output**2)

grads = tape.gradient(loss, thermo_model.trainable_variables)
initial_energy = sum(scalar(ops.sum(v * v)) for v in thermo_model.trainable_variables)

updated_energy = 0.0
learning_rate = 0.01
for v, g in zip(thermo_model.trainable_variables, grads):
    if g is not None:
        updated = v - learning_rate * g
        updated_energy += scalar(ops.sum(updated * updated))

work_done = initial_energy - updated_energy
efficiency = work_done / (initial_energy + 1e-8)

thermodynamics = {
    "initial_energy": initial_energy,
    "final_energy": updated_energy,
    "work_done": work_done,
    "efficiency": efficiency,
}

print("Learning thermodynamics:")
for key, value in thermodynamics.items():
    print(f"{key}: {value:.6f}")

viz_stage("stage_final", locals())
