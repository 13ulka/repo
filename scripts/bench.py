"""Quick compute benchmark: CPU matmul GFLOPS (fp32), memory bandwidth, and GPU (MPS) if available.
Run the same script on each machine to compare. Usage: python scripts/bench.py"""

import os
import platform
import time

import numpy as np


def best_of(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)


def cpu_matmul(size=4096):
    a = np.random.rand(size, size).astype(np.float32)
    b = np.random.rand(size, size).astype(np.float32)
    t = best_of(lambda: a @ b, 3)
    return 2 * size**3 / t / 1e9


def mem_bandwidth(mb=1024):
    a = np.ones(mb * 1024 * 1024 // 8, dtype=np.float64)
    b = np.empty_like(a)
    t = best_of(lambda: np.copyto(b, a))
    return 2 * a.nbytes / t / 1e9  # read + write


def mps_matmul(size=8192):
    try:
        import torch
    except ImportError:
        return None
    if not torch.backends.mps.is_available():
        return None
    a = torch.rand(size, size, device="mps")
    b = torch.rand(size, size, device="mps")

    def run():
        (a @ b).sum().item()

    t = best_of(run, 5)
    return 2 * size**3 / t / 1e9


print(f"machine: {platform.machine()} {platform.system()} | logical CPUs: {os.cpu_count()}")
print(f"CPU fp32 matmul:   {cpu_matmul():8.1f} GFLOPS  (numpy {np.__version__})")
print(f"memory bandwidth:  {mem_bandwidth():8.1f} GB/s    (single-thread copy)")
g = mps_matmul()
print(f"GPU (MPS) matmul:  {g:8.1f} GFLOPS" if g else "GPU (MPS) matmul:  n/a")
