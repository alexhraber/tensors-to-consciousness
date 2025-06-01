import mlx.core as mx
import time

print("🍎 MLX on Apple Silicon Test")
print("=" * 40)

# Check Metal availability
print(f"Metal available: {mx.metal.is_available()}")
print(f"Default device: {mx.default_device()}")

# Basic array operations
print("\n--- Basic Operations ---")
x = mx.array([1, 2, 3, 4, 5])
y = mx.array([2, 3, 4, 5, 6])

print(f"x = {x}")
print(f"y = {y}")
print(f"x + y = {x + y}")
print(f"x * y = {x * y}")

# Matrix operations
print("\n--- Matrix Operations ---")
A = mx.random.normal((100, 100))
B = mx.random.normal((100, 100))

start = time.time()
C = mx.matmul(A, B)
mx.eval(C)  # Force evaluation
elapsed = time.time() - start

print(f"Matrix multiplication (100x100): {elapsed:.6f} seconds")
print(f"Result shape: {C.shape}")

print("\n✅ MLX setup successful!")
