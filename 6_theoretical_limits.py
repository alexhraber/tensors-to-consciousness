# 6_theoretical_limits_fixed.py
import mlx.core as mx
import mlx.nn as nn

print("🌌 THEORETICAL LIMITS & FUNDAMENTAL BOUNDARIES")
print("=" * 50)

# Set global dtype for consistency
DTYPE = mx.float32

# 1. INFORMATION GEOMETRY: The curved space of probability distributions
print("\n--- Information Geometry & Natural Gradients ---")

def kl_divergence(p, q):
    """Kullback-Leibler divergence: D_KL(P||Q)"""
    eps = 1e-8
    p_safe = mx.maximum(p, eps)
    q_safe = mx.maximum(q, eps)
    return mx.sum(p_safe * mx.log(p_safe / q_safe))

def fisher_information_matrix(model, x):
    """Fisher Information Matrix: measures curvature of loss landscape"""
    def log_likelihood(params):
        output = model(x)
        return -mx.sum(output**2) / 2
    
    grad_fn = mx.grad(log_likelihood)
    gradients = grad_fn(model.parameters())
    
    grad_flat = []
    for name, grad in gradients.items():
        if hasattr(grad, 'flatten'):
            grad_flat.append(grad.flatten())
    
    if grad_flat:
        grad_vector = mx.concatenate(grad_flat)
        fisher_approx = mx.outer(grad_vector, grad_vector)
        return fisher_approx
    return None

# Test information geometry
simple_model = nn.Linear(5, 3)
test_data = mx.random.normal((10, 5), dtype=DTYPE)

fisher_matrix = fisher_information_matrix(simple_model, test_data)
if fisher_matrix is not None:
    print(f"Fisher matrix shape: {fisher_matrix.shape}")
    fisher_max = mx.max(fisher_matrix)
    fisher_min = mx.min(fisher_matrix)
    condition_number = fisher_max / (mx.abs(fisher_min) + 1e-8)
    print(f"Fisher matrix condition number: {condition_number:.2f}")

# 2. KOLMOGOROV COMPLEXITY: Algorithmic Information Theory
print("\n--- Algorithmic Information Theory ---")

def estimate_kolmogorov_complexity(data):
    """Estimate Kolmogorov complexity via compression"""
    complexities = {}
    
    # Convert data to binary using mean as threshold (your fix!)
    data_flat = data.flatten()
    threshold = mx.mean(data_flat)
    data_binary = (data_flat > threshold).astype(mx.int32)
    
    # Method 1: Run-length encoding simulation
    def rle_compress(binary_data):
        # Convert to Python list for processing
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
    
    # Method 2: Simplified entropy estimation
    def entropy_compress(binary_data):
        # Count 0s and 1s manually (avoiding mx.unique)
        total = len(binary_data)
        ones = mx.sum(binary_data.astype(DTYPE))
        zeros = total - ones
        
        if ones > 0 and zeros > 0:
            p_ones = ones / total
            p_zeros = zeros / total
            entropy = -(p_ones * mx.log2(p_ones) + p_zeros * mx.log2(p_zeros))
        else:
            entropy = mx.array(0.0, dtype=DTYPE)
        
        return float(entropy * total / 8)  # Convert to "bytes"
    
    complexities['rle'] = rle_compress(data_binary)
    complexities['entropy'] = entropy_compress(data_binary)
    
    return complexities

# Test on different data types
random_data = mx.random.normal((100,), dtype=DTYPE)
structured_data = mx.sin(mx.arange(0, 4*mx.pi, 4*mx.pi/100, dtype=DTYPE))

random_complexity = estimate_kolmogorov_complexity(random_data)
structured_complexity = estimate_kolmogorov_complexity(structured_data)

print(f"Random data complexity: {random_complexity}")
print(f"Structured data complexity: {structured_complexity}")
entropy_ratio = random_complexity['entropy'] / (structured_complexity['entropy'] + 1e-8)
print(f"Compression ratio (random vs structured): {entropy_ratio:.2f}")

# 3. QUANTUM-INSPIRED NEURAL COMPUTATION
print("\n--- Quantum-Inspired Computation ---")

def quantum_superposition_layer(x, n_qubits=4):
    """Quantum-inspired superposition in amplitude space"""
    batch_size, input_dim = x.shape
    
    # Create complex-valued amplitudes
    real_amplitudes = mx.random.normal((batch_size, n_qubits), dtype=DTYPE)
    imag_amplitudes = mx.random.normal((batch_size, n_qubits), dtype=DTYPE)
    
    # Normalize for probability conservation
    amplitude_norms = mx.sqrt(real_amplitudes**2 + imag_amplitudes**2)
    total_norm = mx.sum(amplitude_norms, axis=1, keepdims=True)
    
    real_normalized = real_amplitudes / (total_norm + 1e-8)
    imag_normalized = imag_amplitudes / (total_norm + 1e-8)
    
    # Quantum measurement: collapse to classical state
    probabilities = real_normalized**2 + imag_normalized**2
    qubit_indices = mx.arange(n_qubits, dtype=DTYPE).reshape(1, -1)
    classical_output = mx.sum(probabilities * qubit_indices, axis=1)
    
    return classical_output, probabilities

quantum_input = mx.random.normal((5, 8), dtype=DTYPE)
quantum_output, quantum_probs = quantum_superposition_layer(quantum_input)

print(f"Quantum output shape: {quantum_output.shape}")
print(f"Probability conservation check: {mx.sum(quantum_probs, axis=1)}")
print(f"Quantum coherence measure: {mx.std(quantum_probs):.4f}")

# 4. INTEGRATED INFORMATION THEORY
print("\n--- Integrated Information Theory (Φ) ---")

def compute_phi(network_state, connectivity_matrix):
    """Compute Integrated Information Φ"""
    n_nodes = len(network_state)
    
    # System entropy
    system_entropy = -mx.sum(network_state * mx.log2(network_state + 1e-8))
    
    # Part entropies
    part_entropies = mx.array(0.0, dtype=DTYPE)
    for i in range(n_nodes):
        node_prob = network_state[i]
        if node_prob > 0 and node_prob < 1:
            node_entropy = -(node_prob * mx.log2(node_prob) + 
                           (1-node_prob) * mx.log2(1-node_prob))
            part_entropies += node_entropy
    
    phi = system_entropy - part_entropies
    connectivity_strength = mx.sum(mx.abs(connectivity_matrix))
    
    return float(phi), float(connectivity_strength)

# Test consciousness measures
connected_state = mx.softmax(mx.random.normal((6,), dtype=DTYPE), axis=0)
connected_matrix = mx.random.normal((6, 6), dtype=DTYPE) * 0.8

disconnected_state = mx.softmax(mx.random.normal((6,), dtype=DTYPE), axis=0) 
disconnected_matrix = mx.eye(6, dtype=DTYPE) * 0.1

phi_connected, conn_connected = compute_phi(connected_state, connected_matrix)
phi_disconnected, conn_disconnected = compute_phi(disconnected_state, disconnected_matrix)

print(f"Connected network Φ: {phi_connected:.4f}, connectivity: {conn_connected:.2f}")
print(f"Disconnected network Φ: {phi_disconnected:.4f}, connectivity: {conn_disconnected:.2f}")
consciousness_ratio = phi_connected / (phi_disconnected + 1e-8)
print(f"Consciousness ratio: {consciousness_ratio:.2f}")

# 5. UNIVERSAL APPROXIMATION AT THE LIMIT
print("\n--- Universal Approximation at Infinite Width ---")

def infinite_width_approximation(target_func, x_points, width_schedule):
    """Demonstrate universal approximation as width → ∞"""
    approximation_errors = []
    
    for width in width_schedule:
        class WideNetwork(nn.Module):
            def __init__(self, width):
                super().__init__()
                self.layer1 = nn.Linear(1, width)
                self.layer2 = nn.Linear(width, 1)
            
            def __call__(self, x):
                h = mx.tanh(self.layer1(x))
                return self.layer2(h)
        
        network = WideNetwork(width)
        network_output = network(x_points.reshape(-1, 1))
        target_output = target_func(x_points).reshape(-1, 1)
        
        error = mx.mean((network_output - target_output)**2)
        approximation_errors.append(float(error))
    
    return approximation_errors

def complex_target(x):
    """Complex target function"""
    return mx.sin(3*x) * mx.exp(-x**2) + mx.cos(5*x) * x

x_test = mx.arange(-2, 2, 4/100, dtype=DTYPE)
widths = [10, 50, 100, 500, 1000]

errors = infinite_width_approximation(complex_target, x_test, widths)

print("Universal approximation convergence:")
for width, error in zip(widths, errors):
    print(f"Width {width:4d}: Error = {error:.6f}")

# 6. THERMODYNAMICS OF LEARNING
print("\n--- Thermodynamics of Learning ---")

def compute_learning_thermodynamics(model, data, learning_rate):
    """Compute thermodynamic quantities during learning"""
    # Store initial parameters
    initial_params = {}
    for name, param in model.parameters().items():
        initial_params[name] = param

    def loss_fn(model, x):
        output = model(x)
        return mx.sum(output**2)

    # Compute gradients and update
    grad_fn = mx.grad(loss_fn)
    gradients = grad_fn(model, data)
    
    # Energy = parameter magnitude
    initial_energy = sum(mx.sum(p**2) for p in initial_params.values())
    
    # Simulate parameter update
    updated_energy = mx.array(0.0, dtype=DTYPE)
    for name, param in model.parameters().items():
        if name in gradients:
            updated_param = param - learning_rate * gradients[name]
            updated_energy += mx.sum(updated_param**2)
    
    work_done = initial_energy - updated_energy
    efficiency = work_done / (initial_energy + 1e-8)
    
    return {
        'initial_energy': float(initial_energy),
        'final_energy': float(updated_energy),
        'work_done': float(work_done),
        'efficiency': float(efficiency)
    }

thermo_model = nn.Linear(10, 5)
thermo_data = mx.random.normal((20, 10), dtype=DTYPE)

thermodynamics = compute_learning_thermodynamics(thermo_model, thermo_data, 0.01)

print("Learning thermodynamics:")
for key, value in thermodynamics.items():
    print(f"{key}: {value:.6f}")
