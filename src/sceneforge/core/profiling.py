"""Wall time and memory monitoring for a stage (including worker subprocesses).

On Apple Silicon, GPU allocations live in unified memory and are not always visible
in a process RSS, so we record three numbers:
  - peak RSS of this process tree,
  - peak system-wide used memory (captures MPS/MLX allocations of workers),
  - framework peaks reported by workers themselves (torch.mps / mlx), merged by the runner.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import psutil

GB = 1024**3


@dataclass
class ResourceStats:
    wall_s: float = 0.0
    peak_tree_rss_gb: float = 0.0
    peak_system_used_gb: float = 0.0
    baseline_system_used_gb: float = 0.0
    worker_peaks_gb: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "wall_s": round(self.wall_s, 2),
            "peak_tree_rss_gb": round(self.peak_tree_rss_gb, 3),
            "peak_system_used_gb": round(self.peak_system_used_gb, 3),
            "baseline_system_used_gb": round(self.baseline_system_used_gb, 3),
            "peak_system_delta_gb": round(self.peak_system_used_gb - self.baseline_system_used_gb, 3),
            "worker_peaks_gb": {k: round(v, 3) for k, v in self.worker_peaks_gb.items()},
        }


def _tree_rss(proc: psutil.Process) -> int:
    total = 0
    try:
        procs = [proc, *proc.children(recursive=True)]
    except psutil.Error:
        return 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


class ResourceMonitor:
    """Context manager sampling memory in a background thread."""

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.stats = ResourceStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def _sample(self) -> None:
        proc = psutil.Process()
        while not self._stop.is_set():
            self.stats.peak_tree_rss_gb = max(self.stats.peak_tree_rss_gb, _tree_rss(proc) / GB)
            self.stats.peak_system_used_gb = max(self.stats.peak_system_used_gb, _system_used() / GB)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "ResourceMonitor":
        self._t0 = time.perf_counter()
        self.stats.baseline_system_used_gb = _system_used() / GB
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.stats.wall_s = time.perf_counter() - self._t0


def _system_used() -> int:
    vm = psutil.virtual_memory()
    # total - available is the closest analogue of "memory pressure" usage on macOS.
    return vm.total - vm.available
