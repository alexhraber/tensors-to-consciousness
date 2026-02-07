# 3_neural_theory.py
import mlx.core as mx
import mlx.nn as nn
from utils import viz_stage

print("🧠 NEURAL NETWORK THEORY")
print("=" * 50)

# Set global dtype for consistency
DTYPE = mx.float32

# 1. UNIVERSAL APPROXIMATION: Why networks work
print("\n--- Universal Approximation Theory ---")

def target_function(x):
    """Complex function to approximate: f(x) = sin(2πx) + 0.5*cos(4πx)"""
    return mx.sin(2 * mx.pi * x) + 0.5 * mx.cos(4 * mx.pi * x)

class UniversalApproximator(nn.Module):
    """Simple network demonstrating universal approximation"""
    def __init__(self, hidden_size):
        super().__init__()
        self.layers = [
            nn.Linear(1, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, 1)
        ]
    
    def __call__(self, x):
        # Each layer applies: h = σ(Wx + b)
        # This creates increasingly complex decision boundaries
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation on output
                x = nn.relu(x)  # ReLU creates piecewise linear approximation
        return x

# 2. INFORMATION FLOW: Forward and backward passes
viz_stage("stage_1", locals())
print("\n--- Information Flow Analysis ---")

def analyze_layer_activations(model, x):
    """Track how information flows through layers"""
    activations = []
    current = x
    activations.append(current)
    
    for i, layer in enumerate(model.layers):
        # Linear transformation
        current = layer(current)
        print(f"Layer {i+1} (linear): mean={mx.mean(current):.4f}, std={mx.std(current):.4f}")
        
        # Activation function (except last layer)
        if i < len(model.layers) - 1:
            current = nn.relu(current)
            print(f"Layer {i+1} (ReLU):   mean={mx.mean(current):.4f}, std={mx.std(current):.4f}")
        
        activations.append(current)
    
    return activations

# Test information flow
model = UniversalApproximator(32)
test_input = mx.random.normal((10, 1), dtype=DTYPE)
activations = analyze_layer_activations(model, test_input)

# 3. GRADIENT FLOW: How learning propagates
viz_stage("stage_2", locals())
print("\n--- Gradient Flow Analysis ---")

def loss_function(model, x, y):
    """Mean squared error loss"""
    predictions = model(x)
    return mx.mean((predictions - y)**2)

# Generate training data
x_train = mx.random.uniform(-1, 1, (100, 1), dtype=DTYPE)
y_train = target_function(x_train)

# Compute gradients
grad_fn = mx.grad(loss_function)
gradients = grad_fn(model, x_train, y_train)

print("Gradient magnitudes by parameter:")
for name, grad in gradients.items():
    if hasattr(grad, 'shape'):  # Skip non-tensor gradients
        grad_norm = mx.sqrt(mx.sum(grad**2))
        print(f"{name}: ||∇|| = {grad_norm:.6f}")

# 4. ACTIVATION FUNCTIONS: Nonlinearity is key
viz_stage("stage_3", locals())
print("\n--- Activation Function Analysis ---")

def compare_activations(x):
    """Compare different activation functions"""
    activations = {
        'relu': nn.relu(x),
        'tanh': mx.tanh(x),
        'sigmoid': nn.sigmoid(x),
        'gelu': nn.gelu(x)
    }
    
    for name, activation in activations.items():
        mean_val = mx.mean(activation)
        std_val = mx.std(activation)
        print(f"{name:>8}: mean={mean_val:.4f}, std={std_val:.4f}")

test_x = mx.random.normal((1000,), dtype=DTYPE)
compare_activations(test_x)

viz_stage("stage_final", locals())
