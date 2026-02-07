# 4_advanced_theory.py
import mlx.core as mx
import mlx.nn as nn
from frameworks.mlx.utils import normal, uniform, viz_stage

VIZ_META = {}

print("🔬 ADVANCED COMPUTATIONAL THEORY")
print("=" * 50)

# Set global dtype for consistency
DTYPE = mx.float32

# 1. MANIFOLD LEARNING: Data lives on low-dimensional surfaces
print("\n--- Manifold Learning Theory ---")

def generate_swiss_roll(n_samples):
    """Generate Swiss roll manifold - 2D surface embedded in 3D"""
    t = uniform(0, 4*mx.pi, (n_samples,), dtype=DTYPE)
    height = uniform(-10, 10, (n_samples,), dtype=DTYPE)
    
    x = t * mx.cos(t)
    y = height  
    z = t * mx.sin(t)
    
    return mx.stack([x, y, z], axis=1)

# The Swiss roll is intrinsically 2D but embedded in 3D
swiss_data = generate_swiss_roll(1000)
print(f"Swiss roll shape: {swiss_data.shape}")
print(f"Intrinsic dimension: 2, Ambient dimension: 3")
print(f"Data range: [{mx.min(swiss_data):.2f}, {mx.max(swiss_data):.2f}]")

# 2. INFORMATION BOTTLENECK: Compression vs prediction tradeoff
viz_stage("stage_1", locals())
print("\n--- Information Bottleneck Theory ---")

class InformationBottleneck(nn.Module):
    """Network with explicit bottleneck to study information compression"""
    def __init__(self, input_dim, bottleneck_dim, output_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, bottleneck_dim)
        self.decoder = nn.Linear(bottleneck_dim, output_dim)
        self.bottleneck_dim = bottleneck_dim
    
    def __call__(self, x):
        # Encoder: compress information
        z = self.encoder(x)  # Information bottleneck
        z = mx.tanh(z)  # Bounded activation preserves information bounds
        
        # Decoder: reconstruct from compressed representation
        output = self.decoder(z)
        return output, z  # Return both output and bottleneck

# Information theory: smaller bottleneck = more compression
models = {
    "high_capacity": InformationBottleneck(10, 8, 5),
    "medium_capacity": InformationBottleneck(10, 4, 5), 
    "low_capacity": InformationBottleneck(10, 2, 5)
}

test_data = normal((100, 10), dtype=DTYPE)
for name, model in models.items():
    output, bottleneck = model(test_data)
    compression_ratio = 10 / model.bottleneck_dim
    bottleneck_info = mx.std(bottleneck)  # Information content proxy
    print(f"{name}: compression {compression_ratio:.1f}x, info={bottleneck_info:.4f}")

# 3. ATTENTION MECHANISMS: Query-Key-Value and information routing
viz_stage("stage_2", locals())
print("\n--- Attention Theory ---")

def attention_mechanism(queries, keys, values, temperature=1.0):
    """
    Attention: How to dynamically route information
    Attention(Q,K,V) = softmax(QK^T/√d)V
    """
    # Compute similarity scores
    scores = mx.matmul(queries, keys.T) / mx.sqrt(mx.array(queries.shape[-1], dtype=DTYPE))
    scores = scores / temperature  # Temperature controls sharpness
    
    # Softmax gives probability distribution over keys
    attention_weights = mx.softmax(scores, axis=-1)
    
    # Weighted combination of values
    output = mx.matmul(attention_weights, values)
    
    return output, attention_weights

# Example: 3 queries attending to 4 keys/values
queries = normal((3, 64), dtype=DTYPE)
keys = normal((4, 64), dtype=DTYPE)  
values = normal((4, 64), dtype=DTYPE)

attended_output, weights = attention_mechanism(queries, keys, values)

print(f"Attention weights shape: {weights.shape}")
print(f"Attention weights (row sums): {mx.sum(weights, axis=1)}")  # Should be ~1
print(f"Max attention per query: {mx.max(weights, axis=1)}")

# 4. GRADIENT FLOWS ON MANIFOLDS: Advanced optimization geometry
viz_stage("stage_3", locals())
print("\n--- Riemannian Optimization ---")

def project_to_sphere(x):
    """Project vector onto unit sphere (constraint manifold)"""
    return x / (mx.sqrt(mx.sum(x**2)) + 1e-8)

def riemannian_gradient_step(x, grad, lr=0.01):
    """
    Gradient descent on the sphere manifold
    Must project gradients to tangent space
    """
    # Project gradient to tangent space (orthogonal to x)
    tangent_grad = grad - mx.sum(grad * x) * x
    
    # Take step in tangent space
    x_new = x - lr * tangent_grad
    
    # Project back to manifold (sphere)
    return project_to_sphere(x_new)

# Optimize on sphere manifold
def sphere_objective(x):
    """Function to minimize on unit sphere"""
    target = mx.array([1.0, 0.0, 0.0], dtype=DTYPE)  # Want to reach north pole
    return mx.sum((x - target)**2)

x = project_to_sphere(normal((3,), dtype=DTYPE))  # Start on sphere
print(f"Optimizing on sphere manifold:")
print(f"Initial: obj={sphere_objective(x):.6f}, ||x||={mx.sqrt(mx.sum(x**2)):.6f}")

for i in range(10):
    grad = mx.grad(sphere_objective)(x)
    x = riemannian_gradient_step(x, grad)
    if i % 3 == 0:
        obj_val = sphere_objective(x)
        constraint_violation = abs(mx.sum(x**2) - 1.0)
        print(f"Step {i+1}: obj={obj_val:.6f}, constraint_error={constraint_violation:.8f}")

print(f"Final point: {x}")

viz_stage("stage_final", locals())
