# 0_computational_primitives.py (Keras)
from keras import ops

from utils import DTYPE, normal, scalar, viz_stage

print("🧮 COMPUTATIONAL PRIMITIVES (Keras)")
print("=" * 50)

print("\n--- Tensors as Computational Objects ---")
scalar_t = ops.array(42, dtype=DTYPE)
vector = ops.array([1, 2, 3], dtype=DTYPE)
matrix = ops.array([[1, 2], [3, 4]], dtype=DTYPE)
tensor_3d = normal((2, 3, 4), dtype=DTYPE)

print(f"Scalar (0D): {scalar_t} - represents a single value")
print(f"Vector (1D): {vector} - represents direction & magnitude")
print(f"Matrix (2D): {matrix.shape} - represents linear transformations")
print(f"3D Tensor: {tensor_3d.shape} - represents higher-dimensional data")

viz_stage("stage_1", locals())
print("\n--- Fundamental Operations ---")
a = ops.array([1, 2, 3, 4], dtype=DTYPE)
b = ops.array([2, 3, 4, 5], dtype=DTYPE)
print(f"Element-wise: {a} ⊙ {b} = {a * b}")
print(f"Linear combination: 2*{a} + 3*{b} = {2 * a + 3 * b}")

dot = ops.sum(a * b)
print(f"Dot product: <{a}, {b}> = {dot}")

A = ops.array([[1, 2], [3, 4]], dtype=DTYPE)
B = ops.array([[5, 6], [7, 8]], dtype=DTYPE)
C = ops.matmul(A, B)
print(f"Matrix mult:\n{A} @\n{B} =\n{C}")

viz_stage("stage_2", locals())
print("\n--- Reduction Operations ---")
data = normal((1000,), dtype=DTYPE)
mean = ops.mean(data)
variance = ops.var(data)
max_val = ops.max(data)
norm = ops.sqrt(ops.sum(data * data))

print(f"Mean (central tendency): {scalar(mean):.4f}")
print(f"Variance (spread): {scalar(variance):.4f}")
print(f"L2 norm (magnitude): {scalar(norm):.4f}")
print(f"Max value: {scalar(max_val):.4f}")

viz_stage("stage_final", locals())
