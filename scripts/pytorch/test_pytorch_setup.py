import time

import torch

from utils import DTYPE, normal


def main():
    print("PyTorch Setup Test")
    print("=" * 40)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    print("\n--- Basic Operations ---")
    x = torch.tensor([1, 2, 3, 4, 5], dtype=DTYPE)
    y = torch.tensor([2, 3, 4, 5, 6], dtype=DTYPE)

    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x + y = {x + y}")
    print(f"x * y = {x * y}")

    print("\n--- Matrix Operations ---")
    a = normal((100, 100))
    b = normal((100, 100))

    start = time.time()
    c = torch.matmul(a, b)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"Matrix multiplication (100x100): {elapsed:.6f} seconds")
    print(f"Result shape: {c.shape}")
    print("\nSetup successful!")


if __name__ == "__main__":
    main()
