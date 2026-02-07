import time

import numpy as np

from utils import DTYPE, normal


def main():
    print("NumPy Setup Test")
    print("=" * 40)
    print(f"NumPy version: {np.__version__}")

    print("\n--- Basic Operations ---")
    x = np.array([1, 2, 3, 4, 5], dtype=DTYPE)
    y = np.array([2, 3, 4, 5, 6], dtype=DTYPE)

    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x + y = {x + y}")
    print(f"x * y = {x * y}")

    print("\n--- Matrix Operations ---")
    a = normal((100, 100))
    b = normal((100, 100))

    start = time.time()
    c = np.matmul(a, b)
    elapsed = time.time() - start

    print(f"Matrix multiplication (100x100): {elapsed:.6f} seconds")
    print(f"Result shape: {c.shape}")
    print("\nSetup successful!")


if __name__ == "__main__":
    main()
