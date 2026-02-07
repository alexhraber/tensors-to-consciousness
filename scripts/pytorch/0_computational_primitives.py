# 0_computational_primitives.py (PyTorch)
import torch

from utils import DTYPE, normal, scalar, viz_stage

print("🧮 COMPUTATIONAL PRIMITIVES (PyTorch)")
print("=" * 50)

print("\n--- Tensors as Computational Objects ---")
scalar_t = torch.tensor(42, dtype=DTYPE)
vector = torch.tensor([1, 2, 3], dtype=DTYPE)
matrix = torch.tensor([[1, 2], [3, 4]], dtype=DTYPE)
tensor_3d = normal((2, 3, 4))

print(f"Scalar (0D): {scalar_t} - represents a single value")
print(f"Vector (1D): {vector} - represents direction & magnitude")
print(f"Matrix (2D): {matrix.shape} - represents linear transformations")
print(f"3D Tensor: {tensor_3d.shape} - represents higher-dimensional data")

viz_stage("stage_1", locals())
print("\n--- Fundamental Operations ---")
a = torch.tensor([1, 2, 3, 4], dtype=DTYPE)
b = torch.tensor([2, 3, 4, 5], dtype=DTYPE)
print(f"Element-wise: {a} ⊙ {b} = {a * b}")
print(f"Linear combination: 2*{a} + 3*{b} = {2 * a + 3 * b}")

dot = torch.sum(a * b)
print(f"Dot product: <{a}, {b}> = {dot}")

A = torch.tensor([[1, 2], [3, 4]], dtype=DTYPE)
B = torch.tensor([[5, 6], [7, 8]], dtype=DTYPE)
C = torch.matmul(A, B)
print(f"Matrix mult:\n{A} @\n{B} =\n{C}")

viz_stage("stage_2", locals())
print("\n--- Reduction Operations ---")
data = normal((1000,))
mean = torch.mean(data)
variance = torch.var(data)
max_val = torch.max(data)
norm = torch.sqrt(torch.sum(data * data))

print(f"Mean (central tendency): {scalar(mean):.4f}")
print(f"Variance (spread): {scalar(variance):.4f}")
print(f"L2 norm (magnitude): {scalar(norm):.4f}")
print(f"Max value: {scalar(max_val):.4f}")

viz_stage("stage_final", locals())
