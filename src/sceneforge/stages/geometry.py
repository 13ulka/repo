"""Stage 1 — geometry: joint pose + depth for all keyframes of all shots (MapAnything).

Dynamic pixels (s0b masks) are blacked out before inference and excluded afterwards.
Shots are registered into one coordinate frame by a single joint inference; a per-shot
overlap check flags shots that do not share geometry with the rest (possible failed
registration or genuinely different set dressing in the animation).

Output (work/<scene>/s1_geometry/):
  pred/<frame_id>.npz     depth, conf, valid, K (model res), cam2world (OpenCV)
  inputs/<frame_id>.png   masked model inputs (for colors)
  cameras.json            per frame: K_full, K_model, cam2world, crop params, shot
  points.ply, points.npz  fused colored cloud (npz also has frame index and confidence)
  shots_registration.json overlap/confidence per shot
  proj_*.png, report.html
"""

from __future__ import annotations

import logging
import shutil
from typing import Any

import cv2
import numpy as np

from ..core import geom, report
from ..core.device import confirm_cpu, resolve_device
from ..core.runner import run_worker
from ..core.stage import StageContext
from ..core.workspace import read_json, write_json

log = logging.getLogger("sceneforge")

SHOT_COLORS = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180],
    [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128], [170, 110, 40],
    [128, 0, 0], [170, 255, 195], [128, 128, 0], [0, 0, 128],
], dtype=np.uint8)


def pick_frames(frames: list[dict[str, Any]], excluded: set[str], exclude_shots: set[str], max_frames: int) -> list[dict[str, Any]]:
    keep = [f for f in frames if f["frame_id"] not in excluded and f["shot_id"] not in exclude_shots]
    if len(keep) <= max_frames:
        return keep
    by_shot: dict[str, list[dict[str, Any]]] = {}
    for f in keep:
        by_shot.setdefault(f["shot_id"], []).append(f)
    ratio = max_frames / len(keep)
    out = []
    for fs in by_shot.values():
        n = max(2, int(round(len(fs) * ratio))) if len(fs) >= 2 else len(fs)
        idx = np.linspace(0, len(fs) - 1, num=min(n, len(fs))).round().astype(int)
        out += [fs[i] for i in sorted(set(idx.tolist()))]
    return sorted(out, key=lambda f: f["frame_id"])


def projection_image(points: np.ndarray, colors: np.ndarray, axes: tuple[int, int], size: int = 900) -> np.ndarray:
    p = points[:, axes]
    lo, hi = np.percentile(p, 1, axis=0), np.percentile(p, 99, axis=0)
    span = max(float((hi - lo).max()), 1e-6)
    uv = ((p - lo) / span * (size - 20) + 10).astype(int)
    img = np.full((size, size, 3), 255, np.uint8)
    ok = (uv >= 0).all(1) & (uv < size).all(1)
    img[size - 1 - uv[ok, 1], uv[ok, 0]] = colors[ok][:, ::-1]
    return img


def run(ctx: StageContext) -> dict[str, Any]:
    cfg = ctx.scfg
    ingest, masks = ctx.upstream("s0_ingest"), ctx.upstream("s0b_masks")
    frames = read_json(ingest / "frames.json")
    mask_info = read_json(masks / "masks.json")
    exclude_shots = set(cfg.get("exclude_shots", []))
    sel = pick_frames(frames, set(mask_info["excluded"]), exclude_shots, int(cfg["max_frames"]))
    if len(sel) < 2:
        raise RuntimeError("fewer than 2 usable keyframes for geometry")
    if len(sel) < len(frames):
        ctx.warn(f"geometry uses {len(sel)} of {len(frames)} keyframes (masked-out / excluded shots / max_frames)")

    device = resolve_device(ctx)
    confirm_cpu(ctx, device, est_mps_minutes=max(2.0, len(sel) * 0.05))
    weights = cfg["weights_commercial"] if ctx.cfg["license_profile"] == "commercial" else cfg["weights_noncommercial"]
    job = {
        "frames": [{"id": f["frame_id"], "path": str(ingest / f["file"]), "mask_path": str(masks / "masks" / f"{f['frame_id']}.png")} for f in sel],
        "weights": weights, "device": device, "precision": cfg["precision"],
        "resolution_long_side": int(cfg["resolution_long_side"]), "memory_efficient": bool(cfg["memory_efficient"]),
        "minibatch_size": int(cfg["minibatch_size"]),
    }
    res = run_worker(ctx, "geom", "mapanything_worker.py", job, tag="mapanything")
    job_dir = ctx.out / "_job_mapanything"
    shutil.move(str(job_dir / "pred"), str(ctx.out / "pred"))
    shutil.move(str(job_dir / "inputs"), str(ctx.out / "inputs"))
    if res.get("fp32_fallback"):
        ctx.warn("fp16 produced non-finite values on MPS; the run was repeated in fp32")

    # ---- cameras + fused cloud
    fmeta = {f["frame_id"]: f for f in frames}
    crops = {r["id"]: r["crop"] for r in res["frames"]}
    rng = np.random.default_rng(0)
    all_conf = []
    loaded = []
    for f in sel:
        d = np.load(ctx.out / "pred" / f"{f['frame_id']}.npz")
        loaded.append((f, {k: d[k] for k in d.files}))
        all_conf.append(d["conf"][d["valid"]].astype(np.float32))
    conf_all = np.concatenate(all_conf) if all_conf else np.zeros(1)
    conf_thr = float(np.percentile(conf_all, cfg["confidence_percentile"])) if conf_all.size else 0.0

    cameras, pts, cols, fidx, confs = [], [], [], [], []
    for i, (f, d) in enumerate(loaded):
        K_full = geom.k_model_to_full(d["K"], crops[f["frame_id"]])
        cameras.append({
            "frame_id": f["frame_id"], "shot_id": f["shot_id"], "clip_id": f["clip_id"], "time_s": f["time_s"],
            "frame_index": f["frame_index"], "image": str(ingest / f["file"]), "width": f["width"], "height": f["height"],
            "K_full": K_full.tolist(), "K_model": d["K"].tolist(), "cam2world": d["cam2world"].tolist(), "crop": crops[f["frame_id"]],
        })
        world = geom.backproject(d["depth"], d["K"], d["cam2world"])
        m = d["valid"] & (d["conf"].astype(np.float32) >= conf_thr) & (d["depth"] > 0)
        idx = np.flatnonzero(m.reshape(-1))
        if idx.size > cfg["points_per_frame"]:
            idx = rng.choice(idx, int(cfg["points_per_frame"]), replace=False)
        rgb = cv2.cvtColor(cv2.imread(str(ctx.out / "inputs" / f"{f['frame_id']}.png")), cv2.COLOR_BGR2RGB).reshape(-1, 3)
        pts.append(world.reshape(-1, 3)[idx])
        cols.append(rgb[idx])
        fidx.append(np.full(idx.size, i, np.int32))
        confs.append(d["conf"].reshape(-1)[idx].astype(np.float32))
    P, C, F, CF = np.concatenate(pts), np.concatenate(cols), np.concatenate(fidx), np.concatenate(confs)
    if len(P) < 1000:
        raise RuntimeError(f"only {len(P)} valid points after filtering; geometry failed")
    frame_ids = [c["frame_id"] for c in cameras]
    np.savez_compressed(ctx.out / "points.npz", xyz=P.astype(np.float32), rgb=C, frame=F, conf=CF, frame_ids=np.array(frame_ids))

    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(C.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(str(ctx.out / "points.ply"), pcd)
    write_json(ctx.out / "cameras.json", {"convention": "opencv_cam2world", "model_size": res["model_size"], "weights": res["weights"],
                                           "precision": res["precision"], "frames": cameras})

    # ---- shot registration check
    extent = geom.robust_extent(P)
    voxel = max(extent * float(cfg["voxel_size_rel"]), 1e-6)
    origin = P.min(axis=0)
    shot_of_frame = np.array([fmeta[fid]["shot_id"] for fid in frame_ids])
    shot_ids = sorted(set(shot_of_frame.tolist()))
    keys = {s: geom.voxel_keys_shared_origin(P[shot_of_frame[F] == s], voxel, origin) for s in shot_ids}
    reg = []
    for s in shot_ids:
        others = np.concatenate([keys[o] for o in shot_ids if o != s]) if len(shot_ids) > 1 else np.zeros(0, np.int64)
        overlap = float(np.isin(keys[s], others).mean()) if len(keys[s]) and len(others) else (1.0 if len(shot_ids) == 1 else 0.0)
        m = shot_of_frame[F] == s
        status = "ok" if overlap >= cfg["min_shot_overlap"] else "low_overlap"
        reg.append({"shot_id": s, "frames": int((shot_of_frame == s).sum()), "points": int(m.sum()), "overlap": overlap,
                    "mean_conf": float(CF[m].mean()) if m.any() else 0.0, "status": status})
        if status != "ok":
            ctx.warn(f"shot {s}: only {overlap:.0%} of its geometry overlaps other shots — check proj_*.png; "
                     f"if it is ghosted/misplaced add it to [geometry].exclude_shots")
    write_json(ctx.out / "shots_registration.json", {"voxel": voxel, "extent": extent, "conf_threshold": conf_thr, "shots": reg})

    # ---- visual checks: PCA projections colored by image color and by shot
    mu = P.mean(axis=0)
    _, _, vt = np.linalg.svd((P - mu)[rng.choice(len(P), min(len(P), 200000), replace=False)], full_matrices=False)
    Q = (P - mu) @ vt.T
    shot_idx = np.array([shot_ids.index(s) for s in shot_of_frame[F]])
    shot_col = SHOT_COLORS[shot_idx % len(SHOT_COLORS)]
    for name, axes in (("top", (0, 1)), ("side", (0, 2))):
        cv2.imwrite(str(ctx.out / f"proj_{name}_rgb.png"), projection_image(Q, C, axes))
        cv2.imwrite(str(ctx.out / f"proj_{name}_shots.png"), projection_image(Q, shot_col, axes))
    legend = "".join(
        f'<span style="display:inline-block;padding:2px 6px;margin:2px;background:rgb{tuple(int(c) for c in SHOT_COLORS[i % len(SHOT_COLORS)])}">{s}</span>'
        for i, s in enumerate(shot_ids))

    sections = [
        ("Run", report.table(["frames", "points", "weights", "precision", "inference s", "scene extent (model units)"],
                             [[len(sel), len(P), res["weights"], res["precision"], f"{res['inference_s']:.1f}", f"{extent:.2f}"]])),
        ("Shot registration", report.table(["shot", "frames", "points", "overlap with others", "mean conf", "status"],
                                           [[r["shot_id"], r["frames"], r["points"], f"{r['overlap']:.0%}", f"{r['mean_conf']:.2f}", r["status"]] for r in reg])),
        ("Projection (PCA plane 1-2), colors", report.img_tag("proj_top_rgb.png")),
        ("Projection (PCA plane 1-2), by shot", legend + report.img_tag("proj_top_shots.png")),
        ("Projection (PCA plane 1-3), by shot", report.img_tag("proj_side_shots.png")),
        ("Warnings", report.warnings_html(ctx.warnings)),
    ]
    report.write_html(ctx.out / "report.html", "SceneForge · s1 geometry", sections)
    return {"frames": len(sel), "points": int(len(P)), "shots_low_overlap": [r["shot_id"] for r in reg if r["status"] != "ok"],
            "inference_s": res["inference_s"], "precision": res["precision"]}
