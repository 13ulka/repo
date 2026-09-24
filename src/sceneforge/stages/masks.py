"""Stage 0b — dynamic masks: characters/people/animals segmented with SAM 3.1 (MLX).

Masks mark pixels that must NOT be used as static geometry or texture. They are dilated
to cover soft edges, hair and motion blur.

Output (work/<scene>/s0b_masks/):
  masks/<frame_id>.png   255 = dynamic (excluded)
  masks.json             per-frame masked fraction and detections, excluded frames
  contact_*.jpg, report.html
"""

from __future__ import annotations

import logging
import shutil
from typing import Any

import cv2
import numpy as np

from ..core import report
from ..core.device import resolve_device
from ..core.runner import run_worker
from ..core.stage import StageContext
from ..core.workspace import read_json, write_json

log = logging.getLogger("sceneforge")


def run(ctx: StageContext) -> dict[str, Any]:
    cfg = ctx.scfg
    ingest = ctx.upstream("s0_ingest")
    frames = read_json(ingest / "frames.json")
    if resolve_device(ctx) == "cpu":
        # MLX has no useful CPU fallback for an 870M model; the --device flag applies to torch stages.
        ctx.warn("SAM 3.1 runs on MLX (Apple GPU) regardless of --device cpu")
    log.info("[masks] %d frames x %d prompts, expected ~%.0f min on M-series GPU", len(frames), len(cfg["prompts"]), len(frames) * 1.3 / 60)

    job = {
        "frames": [{"id": f["frame_id"], "path": str(ingest / f["file"])} for f in frames],
        "prompts": list(cfg["prompts"]),
        "score_threshold": float(cfg["score_threshold"]),
        "model": cfg["model"],
        "mode": "union",
    }
    run_worker(ctx, "mlx", "sam3_worker.py", job, tag="sam3")
    job_dir = ctx.out / "_job_sam3"
    detections = read_json(job_dir / "detections.json")

    masks_dir = ctx.out / "masks"
    masks_dir.mkdir()
    k = int(cfg["dilate_px"])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)) if k > 0 else None
    per_frame = []
    excluded = []
    for f in frames:
        m = cv2.imread(str(job_dir / "masks" / f"{f['frame_id']}.png"), cv2.IMREAD_GRAYSCALE)
        if kernel is not None:
            m = cv2.dilate(m, kernel)
        cv2.imwrite(str(masks_dir / f"{f['frame_id']}.png"), m)
        frac = float((m > 127).mean())
        dets = detections.get(f["frame_id"], [])
        per_frame.append({"frame_id": f["frame_id"], "masked_fraction": frac, "detections": dets})
        if frac > cfg["max_masked_fraction"]:
            excluded.append(f["frame_id"])
    shutil.rmtree(job_dir / "masks")  # raw undilated masks are superseded by masks/

    if excluded:
        ctx.warn(f"{len(excluded)} frames are >{cfg['max_masked_fraction']:.0%} masked and will be excluded from geometry")
    no_det_shots = sorted({f["shot_id"] for f in frames} - {f["shot_id"] for f, p in zip(frames, per_frame) if p["detections"]})
    write_json(ctx.out / "masks.json", {"frames": per_frame, "excluded": excluded, "prompts": cfg["prompts"]})

    # visual check: overlay per shot
    sections = [("Summary", report.table(["frames", "with detections", "excluded (too masked)", "mean masked fraction"],
                [[len(frames), sum(1 for p in per_frame if p["detections"]), len(excluded), f"{np.mean([p['masked_fraction'] for p in per_frame]):.3f}"]]))]
    if no_det_shots:
        sections.append(("Shots without any dynamic detection", f"<p>{', '.join(no_det_shots)} — check that characters there are not missed.</p>"))
    for sid in sorted({f["shot_id"] for f in frames}):
        fr = [(f, p) for f, p in zip(frames, per_frame) if f["shot_id"] == sid]
        imgs, labels = [], []
        for f, p in fr:
            img = cv2.imread(str(ingest / f["file"]))
            m = cv2.imread(str(masks_dir / f"{f['frame_id']}.png"), cv2.IMREAD_GRAYSCALE) > 127
            imgs.append(report.overlay_mask(img, m))
            labels.append(f"{f['frame_id']} {p['masked_fraction']:.0%} " + ",".join(sorted({d['label'] for d in p['detections']})))
        sheet = f"contact_{sid}.jpg"
        report.contact_sheet(imgs, labels, ctx.out / sheet)
        sections.append((f"Shot {sid}", report.img_tag(sheet)))
    sections.append(("Warnings", report.warnings_html(ctx.warnings)))
    report.write_html(ctx.out / "report.html", "SceneForge · s0b dynamic masks", sections)
    return {"frames": len(frames), "excluded": len(excluded)}
