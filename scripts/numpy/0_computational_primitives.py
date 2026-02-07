# 0_computational_primitives.py (NumPy)
import numpy as np

from utils import DTYPE, normal, viz_stage

print("🧮 COMPUTATIONAL PRIMITIVES (NumPy)")
print("=" * 50)

print("\n--- Tensors as Computational Objects ---")
scalar = np.array(42, dtype=DTYPE)
vector = np.array([1, 2, 3], dtype=DTYPE)
matrix = np.array([[1, 2], [3, 4]], dtype=DTYPE)
tensor_3d = normal((2, 3, 4))

print(f"Scalar (0D): {scalar} - represents a single value")
print(f"Vector (1D): {vector} - represents direction & magnitude")
print(f"Matrix (2D): {matrix.shape} - represents linear transformations")
print(f"3D Tensor: {tensor_3d.shape} - represents higher-dimensional data")

viz_stage("stage_1", locals())
print("\n--- Fundamental Operations ---")
a = np.array([1, 2, 3, 4], dtype=DTYPE)
b = np.array([2, 3, 4, 5], dtype=DTYPE)
print(f"Element-wise: {a} ⊙ {b} = {a * b}")
print(f"Linear combination: 2*{a} + 3*{b} = {2 * a + 3 * b}")

dot = np.sum(a * b)
print(f"Dot product: <{a}, {b}> = {dot}")

A = np.array([[1, 2], [3, 4]], dtype=DTYPE)
B = np.array([[5, 6], [7, 8]], dtype=DTYPE)
C = np.matmul(A, B)
print(f"Matrix mult:\n{A} @\n{B} =\n{C}")

viz_stage("stage_2", locals())
print("\n--- Reduction Operations ---")
data = normal((1000,))
mean = np.mean(data)
variance = np.var(data)
max_val = np.max(data)
norm = np.sqrt(np.sum(data * data))

print(f"Mean (central tendency): {mean:.4f}")
print(f"Variance (spread): {variance:.4f}")
print(f"L2 norm (magnitude): {norm:.4f}")
print(f"Max value: {max_val:.4f}")

viz_stage("stage_final", locals())
