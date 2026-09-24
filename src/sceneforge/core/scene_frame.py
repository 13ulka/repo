"""Gravity/Manhattan alignment, metric scale from anchors, room box (numpy + open3d)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def fit_plane_ransac(points: np.ndarray, thresh: float, iters: int = 2000, seed: int = 0) -> tuple[np.ndarray, float, np.ndarray]:
    """Return (unit normal n, offset d, inlier mask) for plane n·x + d = 0."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    o3d.utility.random.seed(seed)
    model, inliers = pcd.segment_plane(distance_threshold=float(thresh), ransac_n=3, num_iterations=iters)
    a, b, c, d = model
    n = np.array([a, b, c])
    norm = np.linalg.norm(n)
    mask = np.zeros(len(points), bool)
    mask[np.asarray(inliers, dtype=int)] = True
    return n / norm, d / norm, mask


def estimate_up(floor_pts: np.ndarray | None, cam_centers: np.ndarray, cam_up_dirs: np.ndarray, thresh: float) -> tuple[np.ndarray, str, float | None]:
    """Up vector from the floor plane when available, otherwise from mean camera up (-Y_cv)."""
    cam_up = cam_up_dirs.mean(axis=0)
    cam_up /= np.linalg.norm(cam_up)
    if floor_pts is not None and len(floor_pts) >= 200:
        n, d, inl = fit_plane_ransac(floor_pts, thresh)
        if inl.mean() > 0.3:
            if np.dot(cam_centers.mean(axis=0), n) + d < 0:
                n, d = -n, -d
            # sanity: floor normal should roughly agree with camera up (film cameras are mostly level)
            if np.dot(n, cam_up) > np.cos(np.deg2rad(35)):
                return n, "floor_plane", float(d)
    return cam_up, "camera_up", None


def manhattan_yaw(normals: np.ndarray, up: np.ndarray) -> float:
    """Dominant horizontal direction (radians, mod 90°) of surface normals, in the plane ⟂ up."""
    e1 = np.cross(up, [1.0, 0, 0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(up, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    h = normals - np.outer(normals @ up, up)
    hn = np.linalg.norm(h, axis=1)
    ok = hn > 0.8  # mostly horizontal normals (walls, vertical faces)
    if ok.sum() < 50:
        return 0.0
    ang = np.arctan2(h[ok] @ e2, h[ok] @ e1)
    # average on the circle with 4-fold symmetry
    z = np.exp(4j * ang).mean()
    return float(np.angle(z) / 4.0)


def rotation_from_up_yaw(up: np.ndarray, yaw: float) -> np.ndarray:
    """Rotation R (3x3) mapping model-world vectors into scene axes (x, y horizontal, z = up)."""
    up = up / np.linalg.norm(up)
    e1 = np.cross(up, [1.0, 0, 0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(up, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    x = np.cos(yaw) * e1 + np.sin(yaw) * e2
    y = np.cross(up, x)
    return np.stack([x, y, up])  # rows: scene axes expressed in model world


@dataclass
class AnchorMeasurement:
    object_id: str
    label: str
    quantity: str
    measured: float   # in aligned model units
    expected_m: float
    tolerance_m: float

    @property
    def ratio(self) -> float:
        return self.expected_m / self.measured


def measure_anchor(z: np.ndarray, quantity: str) -> float | None:
    """z = heights above the floor (aligned, model units) of an object's points."""
    if len(z) < 50:
        return None
    if quantity == "height":
        return float(np.percentile(z, 99.5) - max(0.0, np.percentile(z, 0.5)))
    if quantity == "top_height":
        return float(np.percentile(z, 99))
    raise ValueError(f"unknown anchor quantity {quantity}")


def combine_anchors(ms: list[AnchorMeasurement]) -> tuple[float | None, float | None]:
    """Weighted median of scale ratios (weight = 1/relative tolerance) and relative spread."""
    ms = [m for m in ms if m.measured > 1e-6]
    if not ms:
        return None, None
    r = np.array([m.ratio for m in ms])
    w = np.array([m.expected_m / m.tolerance_m for m in ms])
    order = np.argsort(r)
    cw = np.cumsum(w[order])
    med = float(r[order][np.searchsorted(cw, cw[-1] / 2)])
    spread = float((r.max() - r.min()) / med) if len(r) > 1 else 0.0
    return med, spread


def room_box(floor_z: float, xy_points: np.ndarray, ceiling_z: np.ndarray | None, wall_points: np.ndarray | None,
             default_height: float, near: float) -> dict[str, Any]:
    """Axis-aligned room box in scene coordinates (meters, z-up, floor at z=0)."""
    lo = np.percentile(xy_points, 1, axis=0)
    hi = np.percentile(xy_points, 99, axis=0)
    faces: dict[str, Any] = {}
    if ceiling_z is not None and len(ceiling_z) >= 100:
        top = float(np.percentile(ceiling_z, 50))
        faces["ceiling"] = {"source": "observed", "support_points": int(len(ceiling_z))}
    else:
        top = default_height
        faces["ceiling"] = {"source": "inferred", "basis": f"default ceiling height {default_height} m (no ceiling observed)"}
    faces["floor"] = {"source": "observed" if xy_points is not None else "inferred"}
    names = {("x", 0): "wall_x_min", ("x", 1): "wall_x_max", ("y", 0): "wall_y_min", ("y", 1): "wall_y_max"}
    bounds = {"x": (float(lo[0]), float(hi[0])), "y": (float(lo[1]), float(hi[1]))}
    for ai, axis in enumerate(("x", "y")):
        for side in (0, 1):
            pos = bounds[axis][side]
            n = 0 if wall_points is None or len(wall_points) == 0 else int((np.abs(wall_points[:, ai] - pos) < near).sum())
            faces[names[(axis, side)]] = (
                {"source": "observed", "support_points": n} if n >= 200
                else {"source": "inferred", "basis": "extent of observed floor/furniture; wall not visible", "support_points": n}
            )
    return {"min": [bounds["x"][0], bounds["y"][0], float(floor_z)], "max": [bounds["x"][1], bounds["y"][1], float(top)], "faces": faces}
