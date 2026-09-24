"""Stage execution with content-addressed caching, atomic outputs and resource logging.

Cache key = hash(stage name, stage config, code files, upstream stage keys, extra inputs).
If work/<scene>/<stage>/_stage.json has the same key and status "done", the stage is skipped.
Outputs are produced in <stage>.tmp and swapped in only on success, so an interrupted run
never leaves a half-written stage that looks complete.
"""

from __future__ import annotations

import logging
import platform
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .. import __version__
from .hashing import hash_files, hash_json
from .profiling import ResourceMonitor
from .workspace import Workspace, write_json

log = logging.getLogger("sceneforge")


@dataclass
class StageContext:
    ws: Workspace
    name: str
    out: Path                      # temporary output dir (becomes the stage dir on success)
    cfg: dict[str, Any]            # full resolved config
    scfg: dict[str, Any]           # this stage's config section
    assume_yes: bool = False
    warnings: list[str] = field(default_factory=list)
    worker_peaks_gb: dict[str, float] = field(default_factory=dict)

    def warn(self, msg: str) -> None:
        log.warning(msg)
        self.warnings.append(msg)

    def upstream(self, stage: str) -> Path:
        return self.ws.stage_dir(stage)


@dataclass
class StageSpec:
    name: str
    config_key: str
    deps: list[str]
    code_files: list[Path]
    fn: Callable[[StageContext], dict[str, Any]]
    extra_inputs: Callable[[Workspace, dict[str, Any]], Any] | None = None


def compute_key(spec: StageSpec, ws: Workspace, cfg: dict[str, Any]) -> str:
    upstream = {d: ws.require(d)["key"] for d in spec.deps}
    extra = spec.extra_inputs(ws, cfg) if spec.extra_inputs else None
    relevant_cfg = {
        "stage": cfg.get(spec.config_key, {}),
        "license_profile": cfg.get("license_profile"),
        "device": cfg.get("device"),
    }
    return hash_json(
        {
            "stage": spec.name,
            "version": __version__,
            "code": hash_files([p for p in spec.code_files if p.exists()]),
            "cfg": relevant_cfg,
            "upstream": upstream,
            "extra": extra,
        }
    )


def run_stage(spec: StageSpec, ws: Workspace, cfg: dict[str, Any], force: bool = False, assume_yes: bool = False) -> dict[str, Any]:
    key = compute_key(spec, ws, cfg)
    rec = ws.load_record(spec.name)
    if rec and rec.get("status") == "done" and rec.get("key") == key and not force:
        log.info("[%s] cached (key %s…), skipping. Use --force to recompute.", spec.name, key[:10])
        return rec

    final_dir = ws.stage_dir(spec.name)
    tmp_dir = final_dir.with_name(final_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)  # our own leftover from an interrupted run
    tmp_dir.mkdir(parents=True)

    file_handler = logging.FileHandler(tmp_dir / "log.txt")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(file_handler)

    ctx = StageContext(ws=ws, name=spec.name, out=tmp_dir, cfg=cfg, scfg=cfg.get(spec.config_key, {}), assume_yes=assume_yes)
    record: dict[str, Any] = {
        "stage": spec.name,
        "key": key,
        "status": "running",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine()},
    }
    log.info("[%s] start", spec.name)
    try:
        with ResourceMonitor() as mon:
            summary = spec.fn(ctx)
        mon.stats.worker_peaks_gb.update(ctx.worker_peaks_gb)
        record.update(
            status="done",
            finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
            resources=mon.stats.as_dict(),
            summary=summary,
            warnings=ctx.warnings,
        )
    except BaseException as e:
        record.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        write_json(tmp_dir / "_stage.json", record)
        log.removeHandler(file_handler)
        file_handler.close()
        log.error("[%s] failed: %s (partial output kept in %s)", spec.name, e, tmp_dir)
        raise

    write_json(tmp_dir / "_stage.json", record)
    log.removeHandler(file_handler)
    file_handler.close()
    if final_dir.exists():
        shutil.rmtree(final_dir)  # previous output of this same stage, owned by SceneForge
    tmp_dir.replace(final_dir)
    res = record["resources"]
    log.info(
        "[%s] done in %.1fs | peak tree RSS %.2f GB | peak system used +%.2f GB | workers %s",
        spec.name, res["wall_s"], res["peak_tree_rss_gb"], res["peak_system_delta_gb"], res["worker_peaks_gb"] or "-",
    )
    for w in ctx.warnings:
        log.warning("[%s] %s", spec.name, w)
    return record
