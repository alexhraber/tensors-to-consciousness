# 6_theoretical_limits.py (NumPy)
import numpy as np

from utils import DTYPE, finite_diff_grad_dict, init_linear, linear, normal, softmax

print("🌌 THEORETICAL LIMITS & FUNDAMENTAL BOUNDARIES (NumPy)")
print("=" * 50)

print("\n--- Information Geometry & Natural Gradients ---")


def kl_divergence(p, q):
    eps = 1e-8
    p_safe = np.maximum(p, eps)
    q_safe = np.maximum(q, eps)
    return np.sum(p_safe * np.log(p_safe / q_safe))


def fisher_information_matrix(params, x):
    def log_likelihood(p):
        output = linear(p, x)
        return -np.sum(output**2) / 2

    gradients = finite_diff_grad_dict(log_likelihood, params)
    grad_vector = np.concatenate([gradients["weight"].flatten(), gradients["bias"].flatten()])
    return np.outer(grad_vector, grad_vector)


simple_model = init_linear(5, 3)
test_data = normal((10, 5), dtype=DTYPE)
fisher_matrix = fisher_information_matrix(simple_model, test_data)

print(f"Fisher matrix shape: {fisher_matrix.shape}")
fisher_max = np.max(fisher_matrix)
fisher_min = np.min(fisher_matrix)
condition_number = fisher_max / (np.abs(fisher_min) + 1e-8)
print(f"Fisher matrix condition number: {condition_number:.2f}")

print("\n--- Algorithmic Information Theory ---")


def estimate_kolmogorov_complexity(data):
    complexities = {}

    data_flat = data.flatten()
    threshold = np.mean(data_flat)
    data_binary = (data_flat > threshold).astype(np.int32)

    def rle_compress(binary_data):
        binary_list = [int(x) for x in binary_data]
        runs = []
        if len(binary_list) > 0:
            current = binary_list[0]
            count = 1
            for i in range(1, len(binary_list)):
                if binary_list[i] == current:
                    count += 1
                else:
                    runs.append(count)
                    current = binary_list[i]
                    count = 1
            runs.append(count)
        return len(runs)

    def entropy_compress(binary_data):
        total = len(binary_data)
        ones = np.sum(binary_data.astype(DTYPE))
        zeros = total - ones

        if ones > 0 and zeros > 0:
            p_ones = ones / total
            p_zeros = zeros / total
            entropy = -(p_ones * np.log2(p_ones) + p_zeros * np.log2(p_zeros))
        else:
            entropy = np.array(0.0, dtype=DTYPE)

        return float(entropy * total / 8)

    complexities["rle"] = rle_compress(data_binary)
    complexities["entropy"] = entropy_compress(data_binary)
    return complexities


random_data = normal((100,), dtype=DTYPE)
structured_data = np.sin(np.arange(0, 4 * np.pi, 4 * np.pi / 100, dtype=DTYPE))

random_complexity = estimate_kolmogorov_complexity(random_data)
structured_complexity = estimate_kolmogorov_complexity(structured_data)

print(f"Random data complexity: {random_complexity}")
print(f"Structured data complexity: {structured_complexity}")
entropy_ratio = random_complexity["entropy"] / (structured_complexity["entropy"] + 1e-8)
print(f"Compression ratio (random vs structured): {entropy_ratio:.2f}")

print("\n--- Quantum-Inspired Computation ---")


def quantum_superposition_layer(x, n_qubits=4):
    batch_size, input_dim = x.shape

    real_amplitudes = normal((batch_size, n_qubits), dtype=DTYPE)
    imag_amplitudes = normal((batch_size, n_qubits), dtype=DTYPE)

    amplitude_norms = np.sqrt(real_amplitudes**2 + imag_amplitudes**2)
    total_norm = np.sum(amplitude_norms, axis=1, keepdims=True)

    real_normalized = real_amplitudes / (total_norm + 1e-8)
    imag_normalized = imag_amplitudes / (total_norm + 1e-8)

    probabilities = real_normalized**2 + imag_normalized**2
    qubit_indices = np.arange(n_qubits, dtype=DTYPE).reshape(1, -1)
    classical_output = np.sum(probabilities * qubit_indices, axis=1)

    return classical_output, probabilities


quantum_input = normal((5, 8), dtype=DTYPE)
quantum_output, quantum_probs = quantum_superposition_layer(quantum_input)

print(f"Quantum output shape: {quantum_output.shape}")
print(f"Probability conservation check: {np.sum(quantum_probs, axis=1)}")
print(f"Quantum coherence measure: {np.std(quantum_probs):.4f}")

print("\n--- Integrated Information Theory (Φ) ---")


def compute_phi(network_state, connectivity_matrix):
    n_nodes = len(network_state)
    system_entropy = -np.sum(network_state * np.log2(network_state + 1e-8))

    part_entropies = np.array(0.0, dtype=DTYPE)
    for i in range(n_nodes):
        node_prob = network_state[i]
        if node_prob > 0 and node_prob < 1:
            node_entropy = -(node_prob * np.log2(node_prob) + (1 - node_prob) * np.log2(1 - node_prob))
            part_entropies = part_entropies + node_entropy

    phi = system_entropy - part_entropies
    connectivity_strength = np.sum(np.abs(connectivity_matrix))

    return float(phi), float(connectivity_strength)


connected_state = softmax(normal((6,), dtype=DTYPE), axis=0)
connected_matrix = normal((6, 6), dtype=DTYPE) * 0.8

disconnected_state = softmax(normal((6,), dtype=DTYPE), axis=0)
disconnected_matrix = np.eye(6, dtype=DTYPE) * 0.1

phi_connected, conn_connected = compute_phi(connected_state, connected_matrix)
phi_disconnected, conn_disconnected = compute_phi(disconnected_state, disconnected_matrix)

print(f"Connected network Φ: {phi_connected:.4f}, connectivity: {conn_connected:.2f}")
print(f"Disconnected network Φ: {phi_disconnected:.4f}, connectivity: {conn_disconnected:.2f}")
consciousness_ratio = phi_connected / (phi_disconnected + 1e-8)
print(f"Consciousness ratio: {consciousness_ratio:.2f}")

print("\n--- Universal Approximation at Infinite Width ---")


def infinite_width_approximation(target_func, x_points, width_schedule):
    approximation_errors = []

    for width in width_schedule:
        params = {
            "l1": init_linear(1, width),
            "l2": init_linear(width, 1),
        }

        h = np.tanh(linear(params["l1"], x_points.reshape(-1, 1)))
        network_output = linear(params["l2"], h)
        target_output = target_func(x_points).reshape(-1, 1)

        error = np.mean((network_output - target_output) ** 2)
        approximation_errors.append(float(error))

    return approximation_errors


def complex_target(x):
    return np.sin(3 * x) * np.exp(-(x**2)) + np.cos(5 * x) * x


x_test = np.arange(-2, 2, 4 / 100, dtype=DTYPE)
widths = [10, 50, 100, 500, 1000]
errors = infinite_width_approximation(complex_target, x_test, widths)

print("Universal approximation convergence:")
for width, error in zip(widths, errors):
    print(f"Width {width:4d}: Error = {error:.6f}")

print("\n--- Thermodynamics of Learning ---")


def compute_learning_thermodynamics(params, data, learning_rate):
    def loss_fn(p):
        output = linear(p, data)
        return np.sum(output**2)

    gradients = finite_diff_grad_dict(loss_fn, params)
    initial_energy = np.sum(params["weight"] ** 2) + np.sum(params["bias"] ** 2)

    updated_weight = params["weight"] - learning_rate * gradients["weight"]
    updated_bias = params["bias"] - learning_rate * gradients["bias"]
    updated_energy = np.sum(updated_weight**2) + np.sum(updated_bias**2)

    work_done = initial_energy - updated_energy
    efficiency = work_done / (initial_energy + 1e-8)

    return {
        "initial_energy": float(initial_energy),
        "final_energy": float(updated_energy),
        "work_done": float(work_done),
        "efficiency": float(efficiency),
    }


thermo_model = init_linear(10, 5)
thermo_data = normal((20, 10), dtype=DTYPE)
thermodynamics = compute_learning_thermodynamics(thermo_model, thermo_data, 0.01)

print("Learning thermodynamics:")
for key, value in thermodynamics.items():
    print(f"{key}: {value:.6f}")
