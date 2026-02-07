import time

import mlx.core as mx


print("MLX Setup Test")
print("=" * 40)
print(f"Metal available: {mx.metal.is_available()}")
print(f"Default device: {mx.default_device()}")

print("\n--- Basic Operations ---")
x = mx.array([1, 2, 3, 4, 5], dtype=mx.float32)
y = mx.array([2, 3, 4, 5, 6], dtype=mx.float32)

print(f"x = {x}")
print(f"y = {y}")
print(f"x + y = {x + y}")
print(f"x * y = {x * y}")

print("\n--- Matrix Operations ---")
a = mx.random.normal((100, 100))
b = mx.random.normal((100, 100))

start = time.time()
c = mx.matmul(a, b)
mx.eval(c)
elapsed = time.time() - start

print(f"Matrix multiplication (100x100): {elapsed:.6f} seconds")
print(f"Result shape: {c.shape}")
print("\nSetup successful!")
