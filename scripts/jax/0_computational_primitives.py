# 0_computational_primitives.py (JAX)
import jax.numpy as jnp

from utils import DTYPE, normal

print("🧮 COMPUTATIONAL PRIMITIVES (JAX)")
print("=" * 50)

print("\n--- Tensors as Computational Objects ---")
scalar = jnp.array(42, dtype=DTYPE)
vector = jnp.array([1, 2, 3], dtype=DTYPE)
matrix = jnp.array([[1, 2], [3, 4]], dtype=DTYPE)
tensor_3d = normal((2, 3, 4))

print(f"Scalar (0D): {scalar} - represents a single value")
print(f"Vector (1D): {vector} - represents direction & magnitude")
print(f"Matrix (2D): {matrix.shape} - represents linear transformations")
print(f"3D Tensor: {tensor_3d.shape} - represents higher-dimensional data")

print("\n--- Fundamental Operations ---")
a = jnp.array([1, 2, 3, 4], dtype=DTYPE)
b = jnp.array([2, 3, 4, 5], dtype=DTYPE)
print(f"Element-wise: {a} ⊙ {b} = {a * b}")
print(f"Linear combination: 2*{a} + 3*{b} = {2 * a + 3 * b}")

dot = jnp.sum(a * b)
print(f"Dot product: <{a}, {b}> = {dot}")

A = jnp.array([[1, 2], [3, 4]], dtype=DTYPE)
B = jnp.array([[5, 6], [7, 8]], dtype=DTYPE)
C = jnp.matmul(A, B)
print(f"Matrix mult:\n{A} @\n{B} =\n{C}")

print("\n--- Reduction Operations ---")
data = normal((1000,))
mean = jnp.mean(data)
variance = jnp.var(data)
max_val = jnp.max(data)
norm = jnp.sqrt(jnp.sum(data * data))

print(f"Mean (central tendency): {mean:.4f}")
print(f"Variance (spread): {variance:.4f}")
print(f"L2 norm (magnitude): {norm:.4f}")
print(f"Max value: {max_val:.4f}")
