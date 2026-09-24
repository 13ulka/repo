from pathlib import Path

import cv2
import numpy as np
import pytest

from sceneforge.core.config import load_config
from sceneforge.core.stage import StageSpec, run_stage
from sceneforge.core.workspace import Workspace, read_json
from sceneforge.stages import ingest


def make_video(path: Path, fps: float = 24.0) -> None:
    """Two synthetic 'shots': a panning texture, then a different static texture."""
    rng = np.random.default_rng(0)
    tex_a = cv2.GaussianBlur(cv2.resize((rng.random((45, 160, 3)) * 255).astype(np.uint8), (1280, 360), interpolation=cv2.INTER_CUBIC), (0, 0), 2.0)
    tex_b = cv2.GaussianBlur((rng.random((360, 640, 3)) * 255).astype(np.uint8), (0, 0), 1.0)
    tex_b = cv2.rectangle(tex_b, (100, 100), (400, 300), (0, 0, 255), 3)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 360))
    assert vw.isOpened()
    for i in range(72):  # 3 s pan
        vw.write(np.ascontiguousarray(tex_a[:, i * 4 : i * 4 + 640]))
    for i in range(48):  # 2 s static with slight noise
        noise = rng.integers(0, 3, tex_b.shape, dtype=np.uint8)
        vw.write(cv2.add(tex_b, noise))
    vw.release()


@pytest.fixture()
def clip(tmp_path):
    p = tmp_path / "clip.mp4"
    make_video(p)
    return p


def test_ingest_detects_shots_and_keyframes(tmp_path, clip):
    ws = Workspace(tmp_path / "work" / "t")
    from sceneforge.core.hashing import hash_file

    ws.save_scene({"scene": "t", "clips": [{"path": str(clip), "name": clip.name, "sha256": hash_file(clip)}]})
    cfg = load_config()
    spec = StageSpec("s0_ingest", "ingest", [], [Path(ingest.__file__)], ingest.run,
                     lambda w, c: [[x["path"], x["sha256"]] for x in w.load_scene()["clips"]])
    rec = run_stage(spec, ws, cfg)
    assert rec["status"] == "done"
    shots = read_json(ws.stage_dir("s0_ingest") / "shots.json")
    frames = read_json(ws.stage_dir("s0_ingest") / "frames.json")
    assert len(shots) == 2, shots
    pan = [f for f in frames if f["shot_id"] == shots[0]["shot_id"]]
    static = [f for f in frames if f["shot_id"] == shots[1]["shot_id"]]
    # panning shot gets denser keyframes than the static one (adaptive rate)
    assert len(pan) / 3.0 > len(static) / 2.0
    assert len(static) >= 3
    for f in frames:
        assert (ws.stage_dir("s0_ingest") / f["file"]).exists()
    assert (ws.stage_dir("s0_ingest") / "report.html").exists()
    # cache hit: second run skips
    rec2 = run_stage(spec, ws, cfg)
    assert rec2["started"] == rec["started"]
    # force recomputes
    rec3 = run_stage(spec, ws, cfg, force=True)
    assert rec3["key"] == rec["key"]
