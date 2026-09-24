"""Camera and point-cloud helpers shared by stages (numpy only).

Conventions:
  - Camera poses are cam2world 4x4 in OpenCV convention (+X right, +Y down, +Z forward),
    as returned by MapAnything.
  - Intrinsics K are pinhole 3x3 in pixel units of the image they refer to.
"""

from __future__ import annotations

import numpy as np


def model_input_size(w0: int, h0: int, long_side: int, patch: int = 14) -> tuple[int, int]:
    """Target (W, H), both multiples of `patch`, preserving aspect as closely as possible."""
    if w0 >= h0:
        w = (long_side // patch) * patch
        h = max(patch, int(round(h0 * w / w0 / patch)) * patch)
    else:
        h = (long_side // patch) * patch
        w = max(patch, int(round(w0 * h / h0 / patch)) * patch)
    return w, h


def resize_crop_params(w0: int, h0: int, w: int, h: int) -> dict[str, float]:
    """Scale-then-center-crop mapping from original (w0,h0) to (w,h)."""
    s = max(w / w0, h / h0)
    rw, rh = int(round(w0 * s)), int(round(h0 * s))
    ox, oy = (rw - w) // 2, (rh - h) // 2
    return {"scale": s, "resized_w": rw, "resized_h": rh, "crop_x": ox, "crop_y": oy, "w": w, "h": h}


def k_model_to_full(K: np.ndarray, p: dict[str, float]) -> np.ndarray:
    """Convert intrinsics of the cropped/resized model image back to the original frame."""
    s = p["scale"]
    Kf = K.astype(np.float64).copy()
    Kf[0, 0] /= s
    Kf[1, 1] /= s
    Kf[0, 2] = (K[0, 2] + p["crop_x"]) / s
    Kf[1, 2] = (K[1, 2] + p["crop_y"]) / s
    return Kf


def backproject(depth: np.ndarray, K: np.ndarray, cam2world: np.ndarray) -> np.ndarray:
    """Z-depth (H,W) -> world points (H,W,3)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    pc = np.stack([x, y, depth], axis=-1)
    return pc @ cam2world[:3, :3].T + cam2world[:3, 3]


def project(points: np.ndarray, K: np.ndarray, cam2world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points (N,3) -> pixel coords (N,2) and camera z (N,)."""
    w2c = np.linalg.inv(cam2world)
    pc = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    uv = np.stack([K[0, 0] * pc[:, 0] / z + K[0, 2], K[1, 1] * pc[:, 1] / z + K[1, 2]], axis=-1)
    return uv, z


def robust_extent(points: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> float:
    a = np.percentile(points, lo, axis=0)
    b = np.percentile(points, hi, axis=0)
    return float(np.linalg.norm(b - a))


def voxel_keys(points: np.ndarray, voxel: float) -> np.ndarray:
    """Unique int64 voxel keys of points."""
    q = np.floor(points / voxel).astype(np.int64)
    q -= q.min(axis=0) if len(q) else 0
    # 21 bits per axis is plenty for room-scale clouds at 1% voxels
    q = np.clip(q, 0, (1 << 21) - 1)
    return np.unique((q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2])


def voxel_keys_shared_origin(points: np.ndarray, voxel: float, origin: np.ndarray) -> np.ndarray:
    q = np.floor((points - origin) / voxel).astype(np.int64)
    q = np.clip(q, 0, (1 << 21) - 1)
    return np.unique((q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2])


def opencv_to_blender_cam(cam2world: np.ndarray) -> np.ndarray:
    """OpenCV camera (+Y down, +Z forward) -> Blender camera (+Y up, -Z forward)."""
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return cam2world @ flip
