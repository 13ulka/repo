"""Analytic ray-cast renderer of a box room, used to fake worker outputs in tests.

True (metric, z-up) scene:
  room      [0,4] x [0,3] x [0,2.6]
  desk      solid box [2.5,3.7] x [0.2,0.8] x [0,0.75]
  door      rectangle on wall y=3: x in [0.5,1.4], z in [0,2.03]
  character solid box [1.5,1.9] x [1.2,1.6] x [0,1.7]  (dynamic, must be masked)
The "model world" is an unknown similarity of the true frame: p_model = S * R @ p_true + t.
"""

from __future__ import annotations

import numpy as np

ROOM = (np.array([0.0, 0, 0]), np.array([4.0, 3, 2.6]))
DESK = (np.array([2.5, 0.2, 0.0]), np.array([3.7, 0.8, 0.75]))
CHAR = (np.array([1.5, 1.2, 0.0]), np.array([1.9, 1.6, 1.7]))
DOOR_X, DOOR_Z = (0.5, 1.4), (0.0, 2.03)

LABELS = {0: "floor", 1: "wall", 2: "ceiling", 3: "door", 4: "desk", 5: "character"}
COLORS = {0: (140, 100, 60), 1: (200, 190, 170), 2: (240, 240, 240), 3: (90, 60, 40), 4: (60, 120, 170), 5: (220, 60, 60)}

S_MODEL = 0.7
_rng = np.random.default_rng(42)
_q = np.linalg.qr(_rng.normal(size=(3, 3)))[0]
R_MODEL = _q if np.linalg.det(_q) > 0 else -_q
T_MODEL = np.array([0.3, -1.2, 2.0])


def to_model(p: np.ndarray) -> np.ndarray:
    return S_MODEL * p @ R_MODEL.T + T_MODEL


def look_at(pos, target) -> np.ndarray:
    f = np.asarray(target, float) - pos
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([right, down, f], axis=1)
    c2w[:3, 3] = pos
    return c2w


def c2w_true_to_model(c2w: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = R_MODEL @ c2w[:3, :3]
    out[:3, 3] = to_model(c2w[:3, 3][None])[0]
    return out


def _slab(o, d, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - o) / d
        t2 = (hi - o) / d
    tn = np.nanmax(np.minimum(t1, t2), axis=-1)
    tf = np.nanmin(np.maximum(t1, t2), axis=-1)
    hit = (tn <= tf) & (tn > 1e-6)
    return np.where(hit, tn, np.inf)


def render(c2w: np.ndarray, K: np.ndarray, w: int, h: int):
    """Returns z-depth (true meters), label map, color image (RGB)."""
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    dc = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    d = dc @ c2w[:3, :3].T
    o = np.broadcast_to(c2w[:3, 3], d.shape)
    lo, hi = ROOM
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(d > 0, (hi - o) / d, (lo - o) / d)
    t_room = np.nanmin(np.where(tx > 0, tx, np.inf), axis=-1)
    face_axis = np.nanargmin(np.where(tx > 0, tx, np.inf), axis=-1)
    t_desk = _slab(o, d, *DESK)
    t_char = _slab(o, d, *CHAR)
    t = np.minimum(t_room, np.minimum(t_desk, t_char))
    p = o + d * t[..., None]
    label = np.ones((h, w), np.int32)  # wall
    label[(face_axis == 2) & (p[..., 2] < 0.01)] = 0
    label[(face_axis == 2) & (p[..., 2] > 2.59)] = 2
    door = (face_axis == 1) & (p[..., 1] > 2.99) & (p[..., 0] > DOOR_X[0]) & (p[..., 0] < DOOR_X[1]) & (p[..., 2] < DOOR_Z[1])
    label[door] = 3
    label[t == t_desk] = 4
    label[t == t_char] = 5
    img = np.zeros((h, w, 3), np.uint8)
    checker = ((np.floor(p[..., 0] * 4) + np.floor(p[..., 1] * 4) + np.floor(p[..., 2] * 4)) % 2).astype(np.float32)
    for k, c in COLORS.items():
        m = label == k
        img[m] = (np.array(c) * (0.8 + 0.2 * checker[m, None])).astype(np.uint8)
    return t.astype(np.float32), label, img


def cameras() -> list[tuple[str, np.ndarray]]:
    """(shot_id, c2w_true) for 3 shots x 5 frames looking around the room."""
    out = []
    shots = {
        "c0s00": ([(1.0, 0.6, 1.5), (1.3, 0.6, 1.5), (1.6, 0.6, 1.5), (1.9, 0.6, 1.5), (2.2, 0.6, 1.55)], (2.5, 3.0, 1.0)),
        "c0s01": ([(3.4, 2.4, 1.6), (3.2, 2.5, 1.6), (3.0, 2.6, 1.6), (2.8, 2.6, 1.6), (2.6, 2.6, 1.6)], (2.0, 0.2, 0.6)),
        "c1s00": ([(0.5, 1.2, 1.4), (0.5, 1.5, 1.4), (0.5, 1.8, 1.45), (0.6, 2.0, 1.5), (0.7, 2.2, 1.5)], (3.8, 1.2, 1.4)),
    }
    for sid, (poss, target) in shots.items():
        for pos in poss:
            out.append((sid, look_at(np.array(pos, float), target)))
    # a wide view where the door is fully in frame (valid scale anchor)
    out.append(("c0s01", look_at(np.array([1.0, 0.8, 1.1]), (0.95, 3.0, 1.0))))
    # a view with the whole desk in frame (second scale anchor)
    out.append(("c1s00", look_at(np.array([3.1, 1.0, 1.5]), (3.1, 0.5, 0.4))))
    # extra upward-looking frame so the ceiling is observed
    out.append(("c1s00", look_at(np.array([2.0, 1.5, 1.2]), (2.4, 2.0, 2.6))))
    return out


K_FULL = np.array([[1100.0, 0, 960], [0, 1100.0, 540], [0, 0, 1]])
FULL_W, FULL_H = 1920, 1080
