import time

import jax
import jax.numpy as jnp

from utils import DTYPE, normal


def main():
    print("JAX Setup Test")
    print("=" * 40)
    print(f"JAX version: {jax.__version__}")
    print(f"Default backend: {jax.default_backend()}")
    print(f"Available devices: {jax.devices()}")

    print("\n--- Basic Operations ---")
    x = jnp.array([1, 2, 3, 4, 5], dtype=DTYPE)
    y = jnp.array([2, 3, 4, 5, 6], dtype=DTYPE)

    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x + y = {x + y}")
    print(f"x * y = {x * y}")

    print("\n--- Matrix Operations ---")
    a = normal((100, 100))
    b = normal((100, 100))

    start = time.time()
    c = jnp.matmul(a, b)
    c.block_until_ready()
    elapsed = time.time() - start

    print(f"Matrix multiplication (100x100): {elapsed:.6f} seconds")
    print(f"Result shape: {c.shape}")
    print("\nSetup successful!")


if __name__ == "__main__":
    main()
