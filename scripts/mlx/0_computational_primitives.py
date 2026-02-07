# 0_computational_primitives.py
import mlx.core as mx
from utils import normal, viz_stage

VIZ_META = {}

print("🧮 COMPUTATIONAL PRIMITIVES")
print("=" * 50)

# 1. TENSORS: The fundamental data structure
print("\n--- Tensors as Computational Objects ---")
# A tensor is just a multidimensional array, but conceptually it's a way to
# represent data in n-dimensional space
scalar = mx.array(42, dtype=mx.float32)           # 0D tensor (point)
vector = mx.array([1, 2, 3], dtype=mx.float32)      # 1D tensor (line)  
matrix = mx.array([[1, 2], [3, 4]], dtype=mx.float32) # 2D tensor (plane)
tensor_3d = normal((2, 3, 4)) # 3D tensor (volume)

print(f"Scalar (0D): {scalar} - represents a single value")
print(f"Vector (1D): {vector} - represents direction & magnitude")
print(f"Matrix (2D): {matrix.shape} - represents linear transformations")
print(f"3D Tensor: {tensor_3d.shape} - represents higher-dimensional data")

# 2. OPERATIONS: How computation actually happens
viz_stage("stage_1", locals())
print("\n--- Fundamental Operations ---")

# Element-wise operations (parallel computation)
a = mx.array([1, 2, 3, 4], dtype=mx.float32)
b = mx.array([2, 3, 4, 5], dtype=mx.float32)
print(f"Element-wise: {a} ⊙ {b} = {a * b}")  # Hadamard product

# Linear combinations (the heart of linear algebra)
print(f"Linear combination: 2*{a} + 3*{b} = {2*a + 3*b}")

# Dot product (measures similarity/projection)
dot = mx.sum(a * b)
print(f"Dot product: <{a}, {b}> = {dot}")

# Matrix multiplication (composition of linear transformations)
A = mx.array([[1, 2], [3, 4]], dtype=mx.float32)
B = mx.array([[5, 6], [7, 8]], dtype=mx.float32)
C = mx.matmul(A, B)
print(f"Matrix mult:\n{A} @\n{B} =\n{C}")

# 3. REDUCTION OPERATIONS: From many to few
viz_stage("stage_2", locals())
print("\n--- Reduction Operations ---")
data = normal((1000,))

# These operations compress information
mean = mx.mean(data)
variance = mx.var(data)
max_val = mx.max(data)
norm = mx.sqrt(mx.sum(data * data))

print(f"Mean (central tendency): {mean:.4f}")
print(f"Variance (spread): {variance:.4f}")
print(f"L2 norm (magnitude): {norm:.4f}")

viz_stage("stage_final", locals())
