"""sceneforge CLI: one subcommand per stage plus `run`, `status` and `preview`."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from .core.config import load_config, resolve_path
from .core.hashing import hash_file
from .core.stage import StageSpec, run_stage
from .core.workspace import STAGES, Workspace, slugify, write_json
from .stages import geometry, ingest, masks, segment

app = typer.Typer(add_completion=False, no_args_is_help=True, help="SceneForge: film location -> Blender scene")
log = logging.getLogger("sceneforge")
PKG = Path(__file__).resolve().parent


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def _clip_inputs(ws: Workspace, cfg: dict) -> list:
    return [[c["path"], c["sha256"]] for c in ws.load_scene()["clips"]]


SPECS: dict[str, StageSpec] = {
    "s0_ingest": StageSpec("s0_ingest", "ingest", [], [PKG / "stages" / "ingest.py"], ingest.run, _clip_inputs),
    "s0b_masks": StageSpec("s0b_masks", "masks", ["s0_ingest"], [PKG / "stages" / "masks.py", PKG / "workers" / "sam3_worker.py"], masks.run),
    "s1_geometry": StageSpec("s1_geometry", "geometry", ["s0_ingest", "s0b_masks"],
                             [PKG / "stages" / "geometry.py", PKG / "workers" / "mapanything_worker.py", PKG / "core" / "geom.py"], geometry.run),
    "s2_segment": StageSpec("s2_segment", "segment", ["s0_ingest", "s1_geometry"],
                            [PKG / "stages" / "segment.py", PKG / "workers" / "sam3_worker.py", PKG / "core" / "scene_frame.py",
                             PKG / "core" / "geom.py"], segment.run),
}
STAGE_ALIASES = {"ingest": "s0_ingest", "masks": "s0b_masks", "geometry": "s1_geometry", "segment": "s2_segment"}


def _ctx(scene: str, config: Optional[Path], device: Optional[str]) -> tuple[Workspace, dict]:
    overrides = {"device": device} if device else None
    cfg = load_config(config, overrides)
    ws = Workspace.for_scene(scene)
    return ws, cfg


def _register_clips(ws: Workspace, clips: list[Path]) -> None:
    entries = []
    for c in clips:
        p = c.expanduser().resolve()
        if not p.is_file():
            raise typer.BadParameter(f"clip not found: {p}")
        log.info("hashing %s", p.name)
        entries.append({"path": str(p), "name": p.name, "sha256": hash_file(p), "size": p.stat().st_size})
    if ws.scene_file.exists():
        old = ws.load_scene()
        if [e["sha256"] for e in old["clips"]] != [e["sha256"] for e in entries]:
            log.info("clip list changed for scene %s; downstream stages will recompute", ws.root.name)
    ws.save_scene({"scene": ws.root.name, "clips": entries})


def _run(names: list[str], ws: Workspace, cfg: dict, force: bool, yes: bool) -> None:
    write_json(ws.root / "config.resolved.json", cfg)
    for n in names:
        run_stage(SPECS[n], ws, cfg, force=force, assume_yes=yes)


SceneOpt = typer.Option(..., "--scene", "-s", help="Scene name (work/<scene>/)")
ConfigOpt = typer.Option(None, "--config", "-c", help="TOML override for configs/default.toml")
ForceOpt = typer.Option(False, "--force", help="Recompute even if cached")
YesOpt = typer.Option(False, "--yes", "-y", help="Do not ask for confirmations (CPU runs, scale fallback)")
DeviceOpt = typer.Option(None, "--device", help="auto | mps | cpu (torch stages)")
VerboseOpt = typer.Option(False, "--verbose", "-v")


@app.command("ingest")
def cmd_ingest(clips: list[Path], scene: Optional[str] = typer.Option(None, "--scene", "-s"), config: Optional[Path] = ConfigOpt,
               force: bool = ForceOpt, yes: bool = YesOpt, verbose: bool = VerboseOpt):
    """Stage 0: shots + sharp keyframes from one or more clips of the same location."""
    _setup_logging(verbose)
    ws, cfg = _ctx(scene or slugify(clips[0].stem), config, None)
    _register_clips(ws, clips)
    _run(["s0_ingest"], ws, cfg, force, yes)


def _stage_cmd(stage: str, doc: str):
    def cmd(scene: str = SceneOpt, config: Optional[Path] = ConfigOpt, force: bool = ForceOpt, yes: bool = YesOpt,
            device: Optional[str] = DeviceOpt, verbose: bool = VerboseOpt):
        _setup_logging(verbose)
        ws, cfg = _ctx(scene, config, device)
        _run([stage], ws, cfg, force, yes)

    cmd.__doc__ = doc
    return cmd


app.command("masks")(_stage_cmd("s0b_masks", "Stage 0b: SAM 3.1 masks of characters (dynamic pixels)."))
app.command("geometry")(_stage_cmd("s1_geometry", "Stage 1: joint poses + depth + fused cloud (MapAnything)."))
app.command("segment")(_stage_cmd("s2_segment", "Stage 2: objects, scene frame, metric scale, room box."))


@app.command("run")
def cmd_run(clips: list[Path], scene: Optional[str] = typer.Option(None, "--scene", "-s"), config: Optional[Path] = ConfigOpt,
            until: str = typer.Option("segment", help="Last stage: ingest|masks|geometry|segment"),
            force: bool = ForceOpt, yes: bool = YesOpt, device: Optional[str] = DeviceOpt, verbose: bool = VerboseOpt,
            preview_blend: bool = typer.Option(True, "--preview/--no-preview", help="Build preview.blend after segment")):
    """Run the pipeline end to end (cached stages are skipped)."""
    _setup_logging(verbose)
    ws, cfg = _ctx(scene or slugify(clips[0].stem), config, device)
    _register_clips(ws, clips)
    last = STAGE_ALIASES.get(until, until)
    if last not in STAGES:
        raise typer.BadParameter(f"unknown stage {until}")
    names = STAGES[: STAGES.index(last) + 1]
    _run(names, ws, cfg, force, yes)
    if preview_blend and last == "s2_segment":
        _preview(ws, cfg)
    _print_status(ws)


@app.command("status")
def cmd_status(scene: str = SceneOpt):
    """Show stage records (time, memory, summary) for a scene."""
    _setup_logging(False)
    _print_status(Workspace.for_scene(scene))


def _print_status(ws: Workspace) -> None:
    typer.echo(f"scene: {ws.root}")
    for s in STAGES:
        rec = ws.load_record(s)
        tmp = ws.stage_dir(s).with_name(s + ".tmp")
        if not rec:
            typer.echo(f"  {s:<12} {'failed (see ' + str(tmp) + ')' if tmp.exists() else '-'}")
            continue
        r = rec.get("resources", {})
        typer.echo(f"  {s:<12} {rec['status']:<6} {r.get('wall_s', 0):>8.1f}s  sys+{r.get('peak_system_delta_gb', 0):5.1f}GB  "
                   f"workers {r.get('worker_peaks_gb', {})}  {rec.get('summary', '')}")
        for w in rec.get("warnings", []):
            typer.echo(f"      ! {w}")
        rep = ws.stage_dir(s) / "report.html"
        if rep.exists():
            typer.echo(f"      report: {rep}")


def _preview(ws: Workspace, cfg: dict) -> Path:
    ws.require("s2_segment")
    blender = resolve_path(cfg["envs"]["blender"])
    if not blender.exists():
        raise typer.BadParameter(f"Blender not found at {blender} (set [envs].blender)")
    out = ws.root / "preview" / "preview.blend"
    out.parent.mkdir(parents=True, exist_ok=True)
    script = PKG / "blender" / "preview_scene.py"
    cmd = [str(blender), "--background", "--factory-startup", "--python", str(script), "--",
           "--segment", str(ws.stage_dir("s2_segment")), "--out", str(out)]
    log.info("building Blender preview: %s", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0 or not out.exists():
        raise typer.Exit(code=rc or 1)
    log.info("preview: %s", out)
    return out


@app.command("preview")
def cmd_preview(scene: str = SceneOpt, config: Optional[Path] = ConfigOpt, verbose: bool = VerboseOpt):
    """Build work/<scene>/preview/preview.blend: point cloud, cameras with frame backgrounds, object boxes, room box."""
    _setup_logging(verbose)
    ws, cfg = _ctx(scene, config, None)
    _preview(ws, cfg)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
