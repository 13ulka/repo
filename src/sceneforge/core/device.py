"""Device selection and CPU-time warnings."""

from __future__ import annotations

import platform

import typer

from .stage import StageContext

# Rough per-frame CPU cost multipliers vs MPS, used only to print an honest warning.
CPU_SLOWDOWN = {"s0b_masks": 8.0, "s1_geometry": 15.0, "s2_segment": 8.0}


def resolve_device(ctx: StageContext) -> str:
    dev = ctx.cfg.get("device", "auto")
    if dev == "auto":
        return "mps" if (platform.system() == "Darwin" and platform.machine() == "arm64") else "cpu"
    if dev not in ("mps", "cpu"):
        raise ValueError(f"device must be auto|mps|cpu, got {dev!r}")
    return dev


def confirm_cpu(ctx: StageContext, device: str, est_mps_minutes: float) -> None:
    if device != "cpu":
        return
    est = est_mps_minutes * CPU_SLOWDOWN.get(ctx.name, 10.0)
    msg = f"[{ctx.name}] running on CPU. Rough estimate: ~{est:.0f} min (vs ~{est_mps_minutes:.0f} min on MPS)."
    ctx.warn(msg)
    if not ctx.assume_yes and not typer.confirm(msg + " Continue?", default=False):
        raise typer.Abort()
