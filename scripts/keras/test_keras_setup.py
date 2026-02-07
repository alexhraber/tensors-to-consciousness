import time

import keras
from keras import ops

from utils import DTYPE, normal, scalar


def main():
    print("Keras Setup Test")
    print("=" * 40)
    print(f"Keras version: {keras.__version__}")
    print(f"Backend: {keras.backend.backend()}")

    print("\n--- Basic Operations ---")
    x = ops.array([1, 2, 3, 4, 5], dtype=DTYPE)
    y = ops.array([2, 3, 4, 5, 6], dtype=DTYPE)

    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x + y = {x + y}")
    print(f"x * y = {x * y}")

    print("\n--- Matrix Operations ---")
    a = normal((100, 100), dtype=DTYPE)
    b = normal((100, 100), dtype=DTYPE)

    start = time.time()
    c = ops.matmul(a, b)
    _ = ops.convert_to_numpy(c)
    elapsed = time.time() - start

    print(f"Matrix multiplication (100x100): {elapsed:.6f} seconds")
    print(f"Result shape: {c.shape}")
    print("\nSetup successful!")


if __name__ == "__main__":
    main()
