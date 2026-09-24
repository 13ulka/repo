"""Run tool workers inside their isolated Python environments.

Workers are standalone scripts in sceneforge/workers/ that depend only on their own env
(torch/MapAnything in envs/geom, MLX/SAM 3.1 in envs/mlx). They receive a JSON job file
and write results plus `_worker_stats.json` (framework peak memory) into the job's out dir.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import resolve_path
from .stage import StageContext

log = logging.getLogger("sceneforge")
WORKERS_DIR = Path(__file__).resolve().parents[1] / "workers"


class WorkerError(RuntimeError):
    pass


def env_python(ctx: StageContext, env: str) -> Path:
    py = resolve_path(ctx.cfg["envs"][env])
    if not py.exists():
        raise WorkerError(
            f"Python for env '{env}' not found at {py}. Run scripts/setup_mac.sh (or set [envs].{env} in your config)."
        )
    return py


def run_worker(ctx: StageContext, env: str, script: str, job: dict[str, Any], tag: str) -> dict[str, Any]:
    py = env_python(ctx, env)
    job_dir = ctx.out / f"_job_{tag}"
    job_dir.mkdir(parents=True, exist_ok=True)
    job = {**job, "out_dir": str(job_dir)}
    job_file = job_dir / "job.json"
    job_file.write_text(json.dumps(job, indent=2))

    cmd = [str(py), str(WORKERS_DIR / script), str(job_file)]
    log.info("[%s] worker %s (%s): %s", ctx.name, tag, env, " ".join(cmd))
    child_env = os.environ.copy()
    child_env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    child_env["PYTHONUNBUFFERED"] = "1"
    with open(job_dir / "worker.log", "w") as wlog:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=child_env, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            wlog.write(line)
            sys.stdout.write(f"  [{tag}] {line}")
        rc = proc.wait()
    if rc != 0:
        raise WorkerError(f"Worker {script} exited with code {rc}. See {job_dir / 'worker.log'}")

    stats_file = job_dir / "_worker_stats.json"
    if stats_file.exists():
        stats = json.loads(stats_file.read_text())
        if "peak_gb" in stats:
            ctx.worker_peaks_gb[tag] = float(stats["peak_gb"])
    result_file = job_dir / "result.json"
    return json.loads(result_file.read_text()) if result_file.exists() else {}
