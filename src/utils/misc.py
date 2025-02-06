# project/utils/misc.py
# ---------------------
import torch
import random
import numpy as np
import time
import functools

def set_random_seed(seed=42):
    """
    Set the random seed for Python, NumPy, and PyTorch for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def timeit(func):
    """
    A simple decorator to time a function’s execution.
    Usage:
        @timeit
        def my_function(...):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"Function '{func.__name__}' took {elapsed:.4f}s to complete.")
        return result
    return wrapper

if __name__ == "__main__":
    @timeit
    def dummy_operation():
        x = 0
        for i in range(10_000_000):
            x += i
        return x

    res = dummy_operation()
    print("Result:", res)
