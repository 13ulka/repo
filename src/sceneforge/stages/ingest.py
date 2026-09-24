"""Stage 0 — ingest: probe clips, split into shots, pick sharp keyframes adaptively.

Output (work/<scene>/s0_ingest/):
  frames/<frame_id>.png   full-resolution keyframes
  frames.json             per-keyframe metadata (clip, shot, frame index, time, sharpness, motion)
  shots.json              shots per clip with frame ranges and keyframe counts
  clips.json              probe metadata
  contact_*.jpg, report.html
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core import report
from ..core.stage import StageContext
from ..core.workspace import write_json

log = logging.getLogger("sceneforge")


# ---------------------------------------------------------------- probing

def probe_clip(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {path}")
    meta: dict[str, Any] = {
        "path": str(path),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count_header": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    if shutil.which("ffprobe"):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                capture_output=True, text=True, check=True, timeout=60,
            ).stdout
            info = json.loads(out)
            v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
            meta["codec"] = v.get("codec_name")
            meta["pix_fmt"] = v.get("pix_fmt")
            meta["duration"] = float(info.get("format", {}).get("duration", 0.0)) or None
            num, _, den = (v.get("avg_frame_rate") or "0/1").partition("/")
            if float(den or 1) > 0 and float(num) > 0:
                meta["fps"] = float(num) / float(den or 1)
        except (subprocess.SubprocessError, json.JSONDecodeError, ValueError) as e:
            log.warning("ffprobe failed for %s: %s", path, e)
    if not meta["fps"] or meta["fps"] <= 0:
        raise RuntimeError(f"Could not determine fps of {path}")
    return meta


# ---------------------------------------------------------------- shots

def _fnum(tc: Any) -> int:
    return int(tc.frame_num) if hasattr(tc, "frame_num") else int(tc.get_frames())


def detect_shots(path: Path, cfg: dict[str, Any], fps: float) -> list[tuple[int, int]]:
    """Return [start, end) frame ranges of shots."""
    from scenedetect import AdaptiveDetector, ContentDetector, detect

    min_len = max(1, int(round(cfg["min_shot_seconds"] * fps)))
    if cfg["detector"] == "adaptive":
        det = AdaptiveDetector(adaptive_threshold=cfg["adaptive_threshold"], min_scene_len=min_len)
    elif cfg["detector"] == "content":
        det = ContentDetector(threshold=cfg["content_threshold"], min_scene_len=min_len)
    else:
        raise ValueError(f"unknown detector {cfg['detector']}")
    scenes = detect(str(path), det, start_in_scene=True)
    return [(_fnum(a), _fnum(b)) for a, b in scenes]


# ---------------------------------------------------------------- analysis

@dataclass
class FrameStats:
    sharpness: np.ndarray  # Laplacian variance at analysis resolution
    tenengrad: np.ndarray
    luma: np.ndarray
    motion: np.ndarray     # median optical-flow magnitude vs previous frame, fraction of width
    n: int


def analyze_frames(path: Path, width: int) -> FrameStats:
    cap = cv2.VideoCapture(str(path))
    sharp, ten, luma, motion = [], [], [], []
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharp.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
        gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        ten.append(float(np.mean(gx * gx + gy * gy)))
        luma.append(float(g.mean()))
        if prev is None:
            motion.append(0.0)
        else:
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.linalg.norm(flow, axis=2)
            motion.append(float(np.median(mag)) / width)
        prev = g
    cap.release()
    n = len(sharp)
    if n == 0:
        raise RuntimeError(f"No frames decoded from {path}")
    return FrameStats(np.array(sharp), np.array(ten), np.array(luma), np.array(motion), n)


def select_keyframes(stats: FrameStats, start: int, end: int, fps: float, cfg: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
    """Adaptive keyframe selection inside one shot [start, end)."""
    edge = int(cfg["edge_skip_frames"])
    lo, hi = start + edge, end - edge
    info: dict[str, Any] = {"dropped_dark": 0, "dropped_blur": 0}
    if hi - lo < 1:
        lo, hi = start, end
    usable = [i for i in range(lo, hi) if stats.luma[i] > cfg["dark_luma_max"]]
    info["dropped_dark"] = (hi - lo) - len(usable)
    if not usable:
        return [], info

    sharp_combined = stats.sharpness / (np.median(stats.sharpness[usable]) + 1e-9) + stats.tenengrad / (np.median(stats.tenengrad[usable]) + 1e-9)
    med = float(np.median(sharp_combined[usable]))
    sharp_ok = {i for i in usable if sharp_combined[i] >= cfg["sharpness_rel_min"] * med}
    info["dropped_blur"] = len(usable) - len(sharp_ok)

    min_gap = max(1, int(round(fps / cfg["max_fps"])))
    max_gap = max(min_gap, int(round(fps / cfg["min_fps"])))
    selected: list[int] = []
    last = None
    acc = 0.0
    for i in usable:
        acc += stats.motion[i]
        since = (i - last) if last is not None else max_gap
        trigger = last is None or (since >= min_gap and acc >= cfg["motion_step"]) or since >= max_gap
        if not trigger:
            continue
        window = [j for j in range(i - min_gap + 1, i + 1) if j in sharp_ok and (last is None or j - last >= min_gap)]
        if not window:
            continue
        best = max(window, key=lambda j: sharp_combined[j])
        selected.append(best)
        acc = float(np.sum(stats.motion[best + 1 : i + 1]))
        last = best

    need = int(cfg["min_frames_per_shot"])
    if len(selected) < need:
        pool = sorted(sharp_ok or set(usable))
        extra = [pool[int(k)] for k in np.linspace(0, len(pool) - 1, num=min(need, len(pool)))]
        selected = sorted(set(selected) | set(extra))
    info["motion_total"] = float(np.sum(stats.motion[start:end]))
    return sorted(set(selected)), info


def thin_to_cap(per_shot: dict[str, list[int]], cap: int) -> dict[str, list[int]]:
    total = sum(len(v) for v in per_shot.values())
    if total <= cap:
        return per_shot
    ratio = cap / total
    out = {}
    for k, v in per_shot.items():
        keep = max(1, int(round(len(v) * ratio)))
        idx = np.linspace(0, len(v) - 1, num=keep).round().astype(int)
        out[k] = [v[i] for i in sorted(set(idx.tolist()))]
    return out


def extract_frames(path: Path, wanted: dict[int, str], out_dir: Path, ext: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    idx = 0
    remaining = set(wanted)
    while remaining:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            cv2.imwrite(str(out_dir / f"{wanted[idx]}.{ext}"), frame)
            remaining.discard(idx)
        idx += 1
    cap.release()
    if remaining:
        raise RuntimeError(f"Could not decode frames {sorted(remaining)[:5]}… from {path}")


# ---------------------------------------------------------------- stage

def run(ctx: StageContext) -> dict[str, Any]:
    cfg = ctx.scfg
    scene = ctx.ws.load_scene()
    frames_dir = ctx.out / "frames"
    all_frames: list[dict[str, Any]] = []
    all_shots: list[dict[str, Any]] = []
    clips_meta = []
    per_shot: dict[str, list[int]] = {}
    shot_meta: dict[str, dict[str, Any]] = {}
    stats_by_clip: dict[str, tuple[FrameStats, dict[str, Any], Path]] = {}

    for ci, clip in enumerate(scene["clips"]):
        path = Path(clip["path"])
        meta = probe_clip(path)
        meta["clip_id"] = f"c{ci}"
        log.info("[ingest] %s: %dx%d @ %.3f fps", path.name, meta["width"], meta["height"], meta["fps"])
        stats = analyze_frames(path, int(cfg["analysis_width"]))
        meta["frames_decoded"] = stats.n
        meta["duration_decoded"] = stats.n / meta["fps"]
        shots = detect_shots(path, cfg, meta["fps"])
        if not shots:
            shots = [(0, stats.n)]
        shots = [(max(0, a), min(stats.n, b)) for a, b in shots if min(stats.n, b) > a]
        meta["n_shots"] = len(shots)
        log.info("[ingest] %s: %d shots", path.name, len(shots))
        clips_meta.append(meta)
        for si, (a, b) in enumerate(shots):
            sid = f"c{ci}s{si:02d}"
            sel, info = select_keyframes(stats, a, b, meta["fps"], cfg)
            if (b - a) / meta["fps"] < cfg["min_shot_seconds"]:
                sel = []  # flashes / transitions
            per_shot[sid] = sel
            shot_meta[sid] = {"shot_id": sid, "clip_id": f"c{ci}", "clip": path.name, "start_frame": a, "end_frame": b,
                              "start_s": a / meta["fps"], "end_s": b / meta["fps"], **info}
        stats_by_clip[f"c{ci}"] = (stats, meta, path)
    before = sum(len(v) for v in per_shot.values())
    per_shot = thin_to_cap(per_shot, int(cfg["max_keyframes"]))
    after = sum(len(v) for v in per_shot.values())
    if after < before:
        ctx.warn(f"keyframes thinned from {before} to {after} (max_keyframes={cfg['max_keyframes']})")

    ext = cfg["image_format"]
    for cid, (stats, meta, path) in stats_by_clip.items():
        wanted: dict[int, str] = {}
        for sid, sel in per_shot.items():
            if not sid.startswith(cid + "s"):
                continue
            for fi in sel:
                fid = f"{sid}f{fi:05d}"
                wanted[fi] = fid
                all_frames.append({
                    "frame_id": fid, "clip_id": cid, "shot_id": sid, "frame_index": fi,
                    "time_s": fi / meta["fps"], "file": f"frames/{fid}.{ext}",
                    "sharpness": float(stats.sharpness[fi]), "tenengrad": float(stats.tenengrad[fi]),
                    "motion": float(stats.motion[fi]), "width": meta["width"], "height": meta["height"],
                })
        log.info("[ingest] extracting %d keyframes from %s", len(wanted), path.name)
        extract_frames(path, wanted, frames_dir, ext)

    for sid, s in shot_meta.items():
        s["n_keyframes"] = len(per_shot.get(sid, []))
        all_shots.append(s)
    all_frames.sort(key=lambda f: f["frame_id"])

    if len(all_shots) > 1:
        log.info("[ingest] %d shots total: they will be registered into one coordinate frame in s1_geometry", len(all_shots))
    static_shots = [s["shot_id"] for s in all_shots if s["n_keyframes"] and s.get("motion_total", 0) < 0.02]
    if static_shots:
        ctx.warn(f"shots with almost no camera/scene motion (little parallax): {static_shots}")

    write_json(ctx.out / "clips.json", clips_meta)
    write_json(ctx.out / "shots.json", all_shots)
    write_json(ctx.out / "frames.json", all_frames)

    # contact sheets per shot + report
    sections = [("Clips", report.table(["clip", "size", "fps", "duration s", "shots"],
                [[m["clip_id"] + " " + Path(m["path"]).name, f"{m['width']}x{m['height']}", f"{m['fps']:.3f}", f"{m['duration_decoded']:.1f}", m["n_shots"]] for m in clips_meta]))]
    sections.append(("Shots", report.table(["shot", "clip", "start s", "end s", "keyframes", "dropped blur", "dropped dark", "motion"],
                    [[s["shot_id"], s["clip"], f"{s['start_s']:.2f}", f"{s['end_s']:.2f}", s["n_keyframes"], s["dropped_blur"], s["dropped_dark"], f"{s.get('motion_total', 0):.3f}"] for s in all_shots])))
    for s in all_shots:
        fr = [f for f in all_frames if f["shot_id"] == s["shot_id"]]
        if not fr:
            continue
        imgs = [cv2.imread(str(ctx.out / f["file"])) for f in fr]
        sheet = f"contact_{s['shot_id']}.jpg"
        report.contact_sheet(imgs, [f"{f['frame_id']} {f['time_s']:.2f}s" for f in fr], ctx.out / sheet)
        sections.append((f"Shot {s['shot_id']} ({len(fr)} keyframes)", report.img_tag(sheet)))
    sections.append(("Warnings", report.warnings_html(ctx.warnings)))
    report.write_html(ctx.out / "report.html", "SceneForge · s0 ingest", sections)

    return {"clips": len(clips_meta), "shots": len(all_shots), "keyframes": len(all_frames)}
