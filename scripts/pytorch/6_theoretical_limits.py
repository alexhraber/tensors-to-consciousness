# 6_theoretical_limits.py (PyTorch)
import torch
import torch.nn as nn

from utils import DTYPE, normal, scalar, softmax

print("🌌 THEORETICAL LIMITS & FUNDAMENTAL BOUNDARIES (PyTorch)")
print("=" * 50)

print("\n--- Information Geometry & Natural Gradients ---")


def kl_divergence(p, q):
    eps = 1e-8
    p_safe = torch.maximum(p, torch.tensor(eps, dtype=DTYPE))
    q_safe = torch.maximum(q, torch.tensor(eps, dtype=DTYPE))
    return torch.sum(p_safe * torch.log(p_safe / q_safe))


def fisher_information_matrix(model, x):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    output = model(x)
    log_like = -torch.sum(output**2) / 2
    log_like.backward()

    grad_flat = []
    for _, param in model.named_parameters():
        if param.grad is not None:
            grad_flat.append(param.grad.detach().flatten())

    if grad_flat:
        grad_vector = torch.cat(grad_flat)
        return torch.outer(grad_vector, grad_vector)
    return None


simple_model = nn.Linear(5, 3).to(dtype=DTYPE)
test_data = normal((10, 5), dtype=DTYPE)

fisher_matrix = fisher_information_matrix(simple_model, test_data)
if fisher_matrix is not None:
    print(f"Fisher matrix shape: {fisher_matrix.shape}")
    fisher_max = torch.max(fisher_matrix)
    fisher_min = torch.min(fisher_matrix)
    condition_number = fisher_max / (torch.abs(fisher_min) + 1e-8)
    print(f"Fisher matrix condition number: {scalar(condition_number):.2f}")

print("\n--- Algorithmic Information Theory ---")


def estimate_kolmogorov_complexity(data):
    complexities = {}

    data_flat = data.flatten()
    threshold = torch.mean(data_flat)
    data_binary = (data_flat > threshold).to(dtype=torch.int32)

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
        ones = torch.sum(binary_data.to(dtype=DTYPE))
        zeros = total - ones

        if ones > 0 and zeros > 0:
            p_ones = ones / total
            p_zeros = zeros / total
            entropy = -(p_ones * torch.log2(p_ones) + p_zeros * torch.log2(p_zeros))
        else:
            entropy = torch.tensor(0.0, dtype=DTYPE)

        return scalar(entropy * total / 8)

    complexities["rle"] = rle_compress(data_binary)
    complexities["entropy"] = entropy_compress(data_binary)
    return complexities


random_data = normal((100,), dtype=DTYPE)
structured_data = torch.sin(torch.arange(0, 4 * torch.pi, 4 * torch.pi / 100, dtype=DTYPE))

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

    amplitude_norms = torch.sqrt(real_amplitudes**2 + imag_amplitudes**2)
    total_norm = torch.sum(amplitude_norms, dim=1, keepdim=True)

    real_normalized = real_amplitudes / (total_norm + 1e-8)
    imag_normalized = imag_amplitudes / (total_norm + 1e-8)

    probabilities = real_normalized**2 + imag_normalized**2
    qubit_indices = torch.arange(n_qubits, dtype=DTYPE).reshape(1, -1)
    classical_output = torch.sum(probabilities * qubit_indices, dim=1)

    return classical_output, probabilities


quantum_input = normal((5, 8), dtype=DTYPE)
quantum_output, quantum_probs = quantum_superposition_layer(quantum_input)

print(f"Quantum output shape: {quantum_output.shape}")
print(f"Probability conservation check: {torch.sum(quantum_probs, dim=1)}")
print(f"Quantum coherence measure: {scalar(torch.std(quantum_probs)):.4f}")

print("\n--- Integrated Information Theory (Φ) ---")


def compute_phi(network_state, connectivity_matrix):
    n_nodes = len(network_state)
    system_entropy = -torch.sum(network_state * torch.log2(network_state + 1e-8))

    part_entropies = torch.tensor(0.0, dtype=DTYPE)
    for i in range(n_nodes):
        node_prob = network_state[i]
        if node_prob > 0 and node_prob < 1:
            node_entropy = -(node_prob * torch.log2(node_prob) + (1 - node_prob) * torch.log2(1 - node_prob))
            part_entropies = part_entropies + node_entropy

    phi = system_entropy - part_entropies
    connectivity_strength = torch.sum(torch.abs(connectivity_matrix))

    return scalar(phi), scalar(connectivity_strength)


connected_state = softmax(normal((6,), dtype=DTYPE), dim=0)
connected_matrix = normal((6, 6), dtype=DTYPE) * 0.8

disconnected_state = softmax(normal((6,), dtype=DTYPE), dim=0)
disconnected_matrix = torch.eye(6, dtype=DTYPE) * 0.1

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
        class WideNetwork(nn.Module):
            def __init__(self, width):
                super().__init__()
                self.layer1 = nn.Linear(1, width)
                self.layer2 = nn.Linear(width, 1)

            def forward(self, x):
                h = torch.tanh(self.layer1(x))
                return self.layer2(h)

        network = WideNetwork(width).to(dtype=DTYPE)
        network_output = network(x_points.reshape(-1, 1))
        target_output = target_func(x_points).reshape(-1, 1)

        error = torch.mean((network_output - target_output) ** 2)
        approximation_errors.append(scalar(error))

    return approximation_errors


def complex_target(x):
    return torch.sin(3 * x) * torch.exp(-(x**2)) + torch.cos(5 * x) * x


x_test = torch.arange(-2, 2, 4 / 100, dtype=DTYPE)
widths = [10, 50, 100, 500, 1000]

errors = infinite_width_approximation(complex_target, x_test, widths)

print("Universal approximation convergence:")
for width, error in zip(widths, errors):
    print(f"Width {width:4d}: Error = {error:.6f}")

print("\n--- Thermodynamics of Learning ---")


def compute_learning_thermodynamics(model, data, learning_rate):
    initial_params = {name: param.detach().clone() for name, param in model.named_parameters()}

    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    output = model(data)
    loss = torch.sum(output**2)
    loss.backward()

    initial_energy = sum(torch.sum(p**2) for p in initial_params.values())

    updated_energy = torch.tensor(0.0, dtype=DTYPE)
    for name, param in model.named_parameters():
        if param.grad is not None:
            updated_param = param.detach() - learning_rate * param.grad.detach()
            updated_energy = updated_energy + torch.sum(updated_param**2)

    work_done = initial_energy - updated_energy
    efficiency = work_done / (initial_energy + 1e-8)

    return {
        "initial_energy": scalar(initial_energy),
        "final_energy": scalar(updated_energy),
        "work_done": scalar(work_done),
        "efficiency": scalar(efficiency),
    }


thermo_model = nn.Linear(10, 5).to(dtype=DTYPE)
thermo_data = normal((20, 10), dtype=DTYPE)

thermodynamics = compute_learning_thermodynamics(thermo_model, thermo_data, 0.01)

print("Learning thermodynamics:")
for key, value in thermodynamics.items():
    print(f"{key}: {value:.6f}")
