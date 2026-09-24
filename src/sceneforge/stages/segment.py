"""Stage 2 — segmentation & objects.

1. VLM inventory of the location (LM Studio, Qwen3-VL) -> detector class list.
2. SAM 3.1 (MLX) instance masks on the geometry keyframes.
3. Masks are lifted to 3D with s1 depth/poses and associated across frames/shots
   by voxel overlap -> objects with IDs.
4. Scene frame: floor plane -> up, wall normals -> Manhattan yaw, anchors -> metric scale.
   Scene coordinates: meters, Z up, floor at z = 0 (Blender convention).
5. Objects: upright OBB, frames seen, viewing-angle spread, quality 0..1; room box with
   observed/inferred faces.

Output (work/<scene>/s2_segment/):
  inventory.json, classes.json
  objects.json            objects in scene coordinates (+ category structure/prop)
  scene_frame.json        4x4 transform model-world -> scene, scale, anchors used
  cameras_scene.json      cameras in scene frame (OpenCV and Blender conventions)
  room.json               room box with per-face source observed|inferred
  points_scene.ply/.npz, objects_points.npz
  contact_*.jpg, top_objects.png, report.html
"""

from __future__ import annotations

import logging
import re
import zlib
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..core import geom, report, scene_frame
from ..core.runner import run_worker
from ..core.stage import StageContext
from ..core.vlm import VLMClient, VLMUnavailable
from ..core.workspace import read_json, write_json

log = logging.getLogger("sceneforge")

AGGREGATED = {"floor", "wall", "ceiling"}
CHARACTER_WORDS = re.compile(r"\b(person|people|man|woman|girl|boy|child|character|human|face|hair|body|hand)s?\b", re.I)

INVENTORY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["room_type", "structure", "props", "characters"],
    "properties": {
        "room_type": {"type": "string"},
        "structure": {"type": "array", "items": {"type": "string"}},
        "props": {"type": "array", "items": {"type": "string"}},
        "characters": {"type": "array", "items": {"type": "string"}},
    },
}

INVENTORY_PROMPT = """These frames come from an animated film and all show the SAME interior location from different shots.
List what a 3D artist would need to rebuild this location.
- "structure": architectural elements actually visible (e.g. floor, wall, ceiling, window, door, staircase, column).
- "props": static furniture and objects visible (e.g. bed, desk, desk lamp, bookshelf, poster, rug, curtain). Include small objects only if clearly visible.
- "characters": short descriptions of characters/people/animals (they will be masked out, do NOT list them as props).
- "room_type": e.g. "teenage bedroom".
Use short concrete English noun phrases of 1-3 words, singular, usable as prompts for an object detector. No duplicates. Answer with JSON only."""


# ------------------------------------------------------------------ inventory

def choose_inventory_frames(cams: list[dict[str, Any]], frames_meta: dict[str, dict[str, Any]], n: int) -> list[dict[str, Any]]:
    by_shot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cams:
        by_shot[c["shot_id"]].append(c)
    best = [max(v, key=lambda c: frames_meta[c["frame_id"]]["sharpness"]) for v in by_shot.values()]
    if len(best) <= n:
        return best
    idx = np.linspace(0, len(best) - 1, num=n).round().astype(int)
    return [best[i] for i in idx]


def normalize_label(s: str) -> str:
    s = re.sub(r"[^a-z0-9 \-]", "", s.lower()).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def build_classes(cfg: dict[str, Any], inventory: dict[str, Any] | None) -> tuple[list[str], set[str]]:
    structure = [normalize_label(x) for x in cfg["structure_classes"]]
    props: list[str] = []
    if inventory:
        structure += [normalize_label(x) for x in inventory.get("structure", [])]
        props += [normalize_label(x) for x in inventory.get("props", [])]
    props += [normalize_label(x) for x in cfg["default_prop_classes"]]
    out, seen = [], set()
    for x in structure + props:
        if not x or x in seen or CHARACTER_WORDS.search(x):
            continue
        seen.add(x)
        out.append(x)
    out = out[: int(cfg["max_classes"])]
    struct_set = {x for x in structure if x in out}
    return out, struct_set


# ------------------------------------------------------------------ association

@dataclass
class Detection:
    frame: int
    label: str
    score: float
    pts: np.ndarray
    vox: np.ndarray
    pix: int
    truncated: bool = False  # mask touches the image border or a dynamic (character) mask -> maybe partial


@dataclass
class Obj:
    label: str
    dets: list[Detection] = field(default_factory=list)
    vox: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    @property
    def frames(self) -> set[int]:
        return {d.frame for d in self.dets}

    @property
    def complete_dets(self) -> list[Detection]:
        """Detections not cut by the frame border nor touching a character mask."""
        return [d for d in self.dets if not d.truncated and d.pix >= 200]

    @property
    def fully_observed(self) -> bool:
        return bool(self.complete_dets)

    def add(self, d: Detection) -> None:
        self.dets.append(d)
        self.vox = np.union1d(self.vox, d.vox)


def containment(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return len(np.intersect1d(a, b, assume_unique=True)) / min(len(a), len(b))


def associate(dets: list[Detection], thr: float) -> list[Obj]:
    objs: dict[str, list[Obj]] = defaultdict(list)
    for d in sorted(dets, key=lambda d: -d.score):
        cands = objs[d.label]
        best, best_ov = None, 0.0
        for o in cands:
            ov = containment(d.vox, o.vox)
            if ov > best_ov:
                best, best_ov = o, ov
        if best is not None and best_ov >= thr and (d.frame not in best.frames or best_ov > 0.6):
            best.add(d)
        else:
            o = Obj(d.label)
            o.add(d)
            cands.append(o)
    # second pass: merge objects of the same label that ended up overlapping
    out = []
    for label, lst in objs.items():
        merged = True
        while merged:
            merged = False
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    if not (lst[i].frames & lst[j].frames) and containment(lst[i].vox, lst[j].vox) >= max(thr, 0.3):
                        for d in lst[j].dets:
                            lst[i].add(d)
                        lst.pop(j)
                        merged = True
                        break
                if merged:
                    break
        out += lst
    return out


def upright_box(p: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5, snap_tol: float = 0.03) -> dict[str, Any]:
    """Z-up oriented bounding box.

    Yaw = minimum-area robust box over 0..89° (1° steps); snapped to the room (Manhattan) axes
    when that costs less than `snap_tol` extra area — furniture is usually aligned with walls.
    """
    xy = p[:, :2]
    mu = xy.mean(axis=0)
    c = xy - mu

    def box_at(yaw: float):
        R2 = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        loc = c @ R2.T
        lo, hi = np.percentile(loc, lo_pct, axis=0), np.percentile(loc, hi_pct, axis=0)
        return float(np.prod(hi - lo)), lo, hi, R2

    sample = c[np.random.default_rng(0).choice(len(c), min(len(c), 5000), replace=False)] if len(c) > 5000 else c
    areas = []
    for deg in range(90):
        a = np.deg2rad(deg)
        R2 = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        loc = sample @ R2.T
        areas.append(float(np.prod(np.percentile(loc, hi_pct, axis=0) - np.percentile(loc, lo_pct, axis=0))))
    best = int(np.argmin(areas))
    yaw = 0.0 if areas[0] <= areas[best] * (1 + snap_tol) else float(np.deg2rad(best))
    _, lo, hi, R2 = box_at(yaw)
    zlo, zhi = np.percentile(p[:, 2], lo_pct), np.percentile(p[:, 2], hi_pct)
    c_xy = mu + ((lo + hi) / 2) @ R2
    return {"center": [float(c_xy[0]), float(c_xy[1]), float((zlo + zhi) / 2)],
            "size": [float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(zhi - zlo)], "yaw": yaw}


def angle_spread(center: np.ndarray, cam_centers: np.ndarray) -> float:
    if len(cam_centers) < 2:
        return 0.0
    v = cam_centers - center
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    c = np.clip(v @ v.T, -1, 1)
    return float(np.degrees(np.arccos(c.min())))


# ------------------------------------------------------------------ scale

def ask_scale(ctx: StageContext, reason: str, metric_guess: float) -> float:
    msg = (f"Metric scale could not be established automatically: {reason}.\n"
           f"MapAnything's own metric estimate gives scale factor {metric_guess:.3f} (1.0 = trust its metric output).")
    if ctx.assume_yes:
        ctx.warn(msg + " Using MapAnything metric scale (--yes).")
        return metric_guess
    if not sys.stdin.isatty():
        raise RuntimeError(msg + "\nSet [segment] scale = <float> in your config, or rerun interactively / with --yes.")
    import typer

    val = typer.prompt(msg + "\nEnter scale factor (meters per model unit), or press Enter to accept", default=str(metric_guess))
    return float(val)


# ------------------------------------------------------------------ stage

def run(ctx: StageContext) -> dict[str, Any]:
    cfg = ctx.scfg
    ingest, geo = ctx.upstream("s0_ingest"), ctx.upstream("s1_geometry")
    cams = read_json(geo / "cameras.json")["frames"]
    frames_meta = {f["frame_id"]: f for f in read_json(ingest / "frames.json")}

    # 1. inventory
    inventory = None
    if cfg["use_vlm"]:
        client = VLMClient(cfg["vlm_url"], cfg["vlm_model"], cfg["vlm_timeout_s"])
        try:
            client.check()
            inv_frames = choose_inventory_frames(cams, frames_meta, int(cfg["inventory_frames"]))
            imgs = [cv2.imread(c["image"]) for c in inv_frames]
            log.info("[segment] asking VLM %s for an inventory of %d frames", cfg["vlm_model"], len(imgs))
            inventory = client.ask_json(INVENTORY_PROMPT, imgs, INVENTORY_SCHEMA)
            inventory["frames"] = [c["frame_id"] for c in inv_frames]
        except VLMUnavailable as e:
            ctx.warn(f"VLM unavailable, using default class list only. {e}")
        except Exception as e:  # malformed answer etc. — the pipeline continues with defaults
            ctx.warn(f"VLM inventory failed ({type(e).__name__}: {e}); using default class list only")
    classes, struct_set = build_classes(cfg, inventory)
    write_json(ctx.out / "inventory.json", inventory or {})
    write_json(ctx.out / "classes.json", {"classes": classes, "structure": sorted(struct_set)})
    log.info("[segment] %d detector classes: %s", len(classes), ", ".join(classes))

    # 2. SAM 3.1 instances on geometry frames
    crop0 = cams[0]["crop"]
    job = {
        "frames": [{"id": c["frame_id"], "path": c["image"]} for c in cams],
        "prompts": classes, "score_threshold": float(cfg["score_threshold"]), "model": ctx.cfg["masks"]["model"],
        "mode": "instances", "save_size": [crop0["resized_w"], crop0["resized_h"]],
    }
    log.info("[segment] SAM 3.1: %d frames x %d classes (expect ~%.0f min)", len(cams), len(classes), len(cams) * (1.0 + 0.1 * len(classes)) / 60)
    run_worker(ctx, "mlx", "sam3_worker.py", job, tag="sam3_instances")
    job_dir = ctx.out / "_job_sam3_instances"
    det_json = read_json(job_dir / "detections.json")

    # 3. lift to 3D
    pts_all = np.load(geo / "points.npz")["xyz"]
    extent = geom.robust_extent(pts_all)
    voxel = extent * float(cfg["association_voxel_rel"])
    origin = pts_all.min(axis=0) - extent
    rng = np.random.default_rng(0)
    detections: list[Detection] = []
    struct_points: dict[str, list[np.ndarray]] = defaultdict(list)
    for fi, c in enumerate(cams):
        d = np.load(geo / "pred" / f"{c['frame_id']}.npz")
        depth, K, c2w, valid = d["depth"], d["K"], d["cam2world"], d["valid"]
        H, W = depth.shape
        inst = np.load(job_dir / "inst" / f"{c['frame_id']}.npz")
        shape = tuple(int(v) for v in inst["shape"])
        if shape[0] == 0:
            continue
        masks = np.unpackbits(inst["packed"], axis=-1, count=shape[2]).astype(bool)
        cx, cy = c["crop"]["crop_x"], c["crop"]["crop_y"]
        masks = masks[:, cy : cy + H, cx : cx + W]
        world = geom.backproject(depth, K, c2w)
        dyn_ring = cv2.dilate(d["dyn"].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        for k, det in enumerate(det_json[c["frame_id"]]):
            m = masks[k] & valid
            n = int(m.sum())
            if n < 30:
                continue
            p = world[m]
            if det["label"] in AGGREGATED:
                struct_points[det["label"]].append(p[rng.choice(len(p), min(len(p), 4000), replace=False)])
                continue
            sub = p[rng.choice(len(p), min(len(p), 3000), replace=False)]
            mk = masks[k]
            truncated = bool(mk[:2].any() or mk[-2:].any() or mk[:, :2].any() or mk[:, -2:].any() or (mk & dyn_ring).any())
            detections.append(Detection(fi, det["label"], det["score"], sub, geom.voxel_keys_shared_origin(sub, voxel, origin), n, truncated))
    log.info("[segment] %d lifted instance detections", len(detections))

    # 4. associate
    objs = associate(detections, float(cfg["association_iou"]))
    dropped = [o for o in objs if len(o.frames) < int(cfg["min_views"])]
    objs = [o for o in objs if len(o.frames) >= int(cfg["min_views"])]
    if dropped:
        ctx.warn(f"{len(dropped)} single-view detections discarded (min_views={cfg['min_views']})")

    # 5. scene frame
    cam_c2w = np.array([c["cam2world"] for c in cams])
    cam_centers = cam_c2w[:, :3, 3]
    cam_up = -cam_c2w[:, :3, 1]
    floor = np.concatenate(struct_points["floor"]) if struct_points["floor"] else None
    walls = np.concatenate(struct_points["wall"]) if struct_points["wall"] else None
    ceiling = np.concatenate(struct_points["ceiling"]) if struct_points["ceiling"] else None
    up, up_src, _ = scene_frame.estimate_up(floor, cam_centers, cam_up, thresh=extent * 0.01)
    if up_src != "floor_plane":
        ctx.warn("floor plane not found reliably; 'up' taken from average camera orientation")
    import open3d as o3d

    normal_src = walls if walls is not None and len(walls) > 500 else pts_all[rng.choice(len(pts_all), min(len(pts_all), 200000), replace=False)]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(normal_src.astype(np.float64)))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=extent * 0.03, max_nn=30))
    yaw = scene_frame.manhattan_yaw(np.asarray(pcd.normals), up)
    R = scene_frame.rotation_from_up_yaw(up, yaw)
    z_all = pts_all @ R[2]
    floor_z = float(np.median(floor @ R[2])) if floor is not None and up_src == "floor_plane" else float(np.percentile(z_all, 1))

    # 6. scale from anchors
    anchors_cfg = cfg.get("anchors", {})
    measurements: list[scene_frame.AnchorMeasurement] = []
    for oi, o in enumerate(objs):
        a = anchors_cfg.get(o.label)
        if not a:
            continue
        if not o.fully_observed:
            log.info("[segment] anchor candidate %s_%d skipped: never fully in frame", o.label, oi)
            continue
        # occlusion/partial views can only shrink the measurement -> take the max over complete views
        vals = [scene_frame.measure_anchor(d.pts @ R[2] - floor_z, a[0]) for d in o.complete_dets]
        vals = [v for v in vals if v]
        val = max(vals) if vals else None
        if val and val > 0:
            measurements.append(scene_frame.AnchorMeasurement(f"{o.label}_{oi}", o.label, a[0], val, float(a[1]), float(a[2])))
    metric_guess = 1.0  # MapAnything outputs metric scale by design
    scale_mode = cfg["scale"]
    if isinstance(scale_mode, (int, float)):
        scale, scale_src = float(scale_mode), "config"
    else:
        s_est, spread = scene_frame.combine_anchors(measurements)
        if s_est is not None and len(measurements) >= 2 and spread <= cfg["max_anchor_spread"]:
            scale, scale_src = s_est, "anchors"
        elif s_est is not None and len(measurements) == 1:
            scale, scale_src = s_est, "single_anchor"
            agree = abs(np.log(s_est / metric_guess)) < np.log(1.25)
            ctx.warn(f"only one scale anchor ({measurements[0].object_id}); scale {s_est:.3f} "
                     + ("agrees with" if agree else "DISAGREES with") + f" MapAnything metric ({metric_guess:.2f}) — check room size in the report")
        else:
            reason = ("no anchors found" if s_est is None else
                      f"{len(measurements)} anchor(s), spread {spread:.0%}, estimate {s_est:.3f}")
            scale = ask_scale(ctx, reason, metric_guess)
            scale_src = "user_or_metric_fallback"

    # 7. transform: p_scene = scale * R (p - o)
    xy_center = np.median((pts_all @ R[:2].T), axis=0)
    o_model = R.T @ np.array([xy_center[0], xy_center[1], floor_z])
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = -scale * R @ o_model

    def to_scene(p: np.ndarray) -> np.ndarray:
        return p @ T[:3, :3].T + T[:3, 3]

    # cameras in scene frame
    cams_scene = []
    for c in cams:
        c2w = np.array(c["cam2world"])
        c2w_s = np.eye(4)
        c2w_s[:3, :3] = R @ c2w[:3, :3]
        c2w_s[:3, 3] = to_scene(c2w[:3, 3][None])[0]
        cams_scene.append({**{k: c[k] for k in ("frame_id", "shot_id", "clip_id", "time_s", "frame_index", "image", "width", "height", "K_full")},
                           "cam2world_opencv": c2w_s.tolist(), "matrix_world_blender": geom.opencv_to_blender_cam(c2w_s).tolist()})
    write_json(ctx.out / "cameras_scene.json", {"units": "meters", "up": "+Z", "frames": cams_scene})

    # 8. objects in scene coords
    cam_centers_scene = to_scene(cam_centers)
    objects = []
    obj_points = {}
    label_count: dict[str, int] = defaultdict(int)
    for o in sorted(objs, key=lambda o: (o.label, -len(o.frames))):
        label_count[o.label] += 1
        oid = f"{re.sub(r'[^a-z0-9]+', '_', o.label)}_{label_count[o.label]:02d}"
        p = to_scene(np.concatenate([d.pts for d in o.dets]))
        box = upright_box(p)
        fr = sorted(o.frames)
        spread = angle_spread(np.array(box["center"]), cam_centers_scene[fr])
        shots = sorted({cams[i]["shot_id"] for i in fr})
        mean_score = float(np.mean([d.score for d in o.dets]))
        q = (0.4 * min(1.0, len(fr) / cfg["good_coverage_views"]) + 0.4 * min(1.0, spread / cfg["good_coverage_angle_deg"]) + 0.2 * mean_score)
        objects.append({
            "id": oid, "label": o.label, "category": "structure" if o.label in struct_set else "prop",
            "source": "observed", "bbox": box, "frames": [cams[i]["frame_id"] for i in fr], "shots": shots,
            "n_views": len(fr), "view_angle_spread_deg": spread, "mean_score": mean_score, "quality": round(q, 3),
            "fully_observed": o.fully_observed,
            "coverage": "good" if (len(fr) >= cfg["good_coverage_views"] and spread >= cfg["good_coverage_angle_deg"]) else "poor",
            "best_frame": cams[max(o.dets, key=lambda d: d.score * d.pix).frame]["frame_id"],
        })
        obj_points[oid] = p.astype(np.float32)
    np.savez_compressed(ctx.out / "objects_points.npz", **obj_points)

    # 9. room box
    P_s = to_scene(pts_all)
    floor_s = to_scene(floor) if floor is not None else P_s[P_s[:, 2] < 0.1]
    room = scene_frame.room_box(
        0.0, np.concatenate([floor_s[:, :2], P_s[P_s[:, 2] < 0.5][:, :2]]) if len(floor_s) else P_s[:, :2],
        to_scene(ceiling)[:, 2] if ceiling is not None else None,
        to_scene(walls) if walls is not None else None,
        default_height=2.6, near=0.08,
    )
    write_json(ctx.out / "room.json", room)

    write_json(ctx.out / "scene_frame.json", {
        "T_model_to_scene": T.tolist(), "scale": scale, "scale_source": scale_src, "up_source": up_src,
        "manhattan_yaw_rad": yaw, "anchors": [m.__dict__ | {"ratio": m.ratio} for m in measurements],
    })
    write_json(ctx.out / "objects.json", {"units": "meters", "objects": objects,
                                          "structure_points": {k: int(sum(len(x) for x in v)) for k, v in struct_points.items()}})
    cols = np.load(geo / "points.npz")["rgb"]
    pcd_s = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P_s.astype(np.float64)))
    pcd_s.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(str(ctx.out / "points_scene.ply"), pcd_s)
    np.savez_compressed(ctx.out / "points_scene.npz", xyz=P_s.astype(np.float32), rgb=cols.astype(np.uint8))

    # 10. report
    _visuals(ctx, cams, det_json, job_dir, objects, P_s, cols, room)
    ceiling_h = room["max"][2]
    sections = [
        ("Scene frame", report.table(["up from", "scale", "scale source", "anchors", "room size (m)", "ceiling"],
                                     [[up_src, f"{scale:.3f}", scale_src, len(measurements),
                                       f"{room['max'][0]-room['min'][0]:.2f} x {room['max'][1]-room['min'][1]:.2f} x {ceiling_h:.2f}",
                                       room["faces"]["ceiling"]["source"]]])),
        ("Anchors", report.table(["object", "quantity", "measured (model)", "expected m", "ratio"],
                                 [[m.object_id, m.quantity, f"{m.measured:.3f}", m.expected_m, f"{m.ratio:.3f}"] for m in measurements])),
        ("Inventory (VLM)", f"<pre>{report.html.escape(str(inventory))}</pre>" if inventory else "<p>VLM not used</p>"),
        ("Objects", report.table(["id", "category", "views", "shots", "angle°", "quality", "coverage", "size m (x,y,z)"],
                                 [[o["id"], o["category"], o["n_views"], len(o["shots"]), f"{o['view_angle_spread_deg']:.0f}", o["quality"], o["coverage"],
                                   " x ".join(f"{v:.2f}" for v in o["bbox"]["size"])] for o in objects])),
        ("Top view (objects)", report.img_tag("top_objects.png")),
    ]
    for sid in sorted({c["shot_id"] for c in cams}):
        sections.append((f"Shot {sid} instances", report.img_tag(f"contact_{sid}.jpg")))
    sections.append(("Warnings", report.warnings_html(ctx.warnings)))
    report.write_html(ctx.out / "report.html", "SceneForge · s2 segmentation & objects", sections)
    return {"classes": len(classes), "objects": len(objects), "good_coverage": sum(o["coverage"] == "good" for o in objects),
            "scale": scale, "scale_source": scale_src, "up_source": up_src}


def _visuals(ctx, cams, det_json, job_dir, objects, P_s, cols, room) -> None:
    rng = np.random.default_rng(1)
    palette = (rng.random((256, 3)) * 200 + 55).astype(np.uint8)
    frame_to_objs = defaultdict(list)
    for i, o in enumerate(objects):
        for f in o["frames"]:
            frame_to_objs[f].append(i)
    for sid in sorted({c["shot_id"] for c in cams}):
        imgs, labels = [], []
        for c in [c for c in cams if c["shot_id"] == sid]:
            img = cv2.imread(c["image"])
            inst = np.load(job_dir / "inst" / f"{c['frame_id']}.npz")
            shape = tuple(int(v) for v in inst["shape"])
            small = cv2.resize(img, (shape[2], shape[1])) if shape[0] else cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
            if shape[0]:
                masks = np.unpackbits(inst["packed"], axis=-1, count=shape[2]).astype(bool)
                for k, det in enumerate(det_json[c["frame_id"]]):
                    color = palette[zlib.crc32(det["label"].encode()) % 256]
                    small = report.overlay_mask(small, masks[k], tuple(int(x) for x in color), 0.45)
            imgs.append(small)
            labels.append(f"{c['frame_id']} " + ",".join(sorted({d['label'] for d in det_json[c['frame_id']]}))[:40])
        report.contact_sheet(imgs, labels, ctx.out / f"contact_{sid}.jpg", cols=5)

    # top view with object boxes (meters)
    size = 1000
    lo = np.array(room["min"][:2]) - 0.3
    hi = np.array(room["max"][:2]) + 0.3
    s = (size - 40) / max(float((hi - lo).max()), 1e-6)

    def px(xy):
        q = (np.asarray(xy) - lo) * s + 20
        return np.stack([q[..., 0], size - q[..., 1]], axis=-1).astype(int)

    img = np.full((size, size, 3), 255, np.uint8)
    sel = rng.choice(len(P_s), min(len(P_s), 300000), replace=False)
    uv = px(P_s[sel, :2])
    ok = (uv >= 0).all(1) & (uv < size).all(1)
    img[uv[ok, 1], uv[ok, 0]] = cols[sel][ok][:, ::-1]
    cv2.rectangle(img, tuple(px(room["min"][:2])), tuple(px(room["max"][:2])), (0, 0, 0), 2)
    for i, o in enumerate(objects):
        b = o["bbox"]
        cx, cy = b["center"][:2]
        sx, sy = b["size"][0] / 2, b["size"][1] / 2
        ca, sa = np.cos(b["yaw"]), np.sin(b["yaw"])
        corners = np.array([[cx + ca * dx - sa * dy, cy + sa * dx + ca * dy] for dx, dy in ((-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy))])
        color = (0, 150, 0) if o["coverage"] == "good" else (0, 0, 220)
        cv2.polylines(img, [px(corners)], True, color, 2)
        cv2.putText(img, o["id"], tuple(px([cx, cy])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(ctx.out / "top_objects.png"), img)
