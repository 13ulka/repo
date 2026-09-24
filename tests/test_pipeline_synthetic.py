"""End-to-end s0b -> s1 -> s2 on a synthetic room with fake workers (no MLX/torch needed)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import synthetic_room as sr
from sceneforge.core import geom
from sceneforge.core.config import load_config
from sceneforge.core.stage import StageSpec, run_stage
from sceneforge.core.workspace import Workspace, read_json, write_json
from sceneforge.stages import geometry, masks, segment

MODEL_W, MODEL_H = geom.model_input_size(sr.FULL_W, sr.FULL_H, 518)
CROP = geom.resize_crop_params(sr.FULL_W, sr.FULL_H, MODEL_W, MODEL_H)
RW, RH = CROP["resized_w"], CROP["resized_h"]
K_RES = sr.K_FULL.copy()
K_RES[:2] *= CROP["scale"]
K_MODEL = K_RES.copy()
K_MODEL[0, 2] -= CROP["crop_x"]
K_MODEL[1, 2] -= CROP["crop_y"]


def frame_id(i, sid):
    return f"{sid}f{i:05d}"


@pytest.fixture(scope="module")
def renders():
    out = {}
    for i, (sid, c2w) in enumerate(sr.cameras()):
        depth, label, img = sr.render(c2w, K_RES, RW, RH)
        out[frame_id(i, sid)] = {"sid": sid, "c2w": c2w, "depth": depth, "label": label, "img": img, "i": i}
    return out


def fake_ingest(renders):
    def run(ctx):
        (ctx.out / "frames").mkdir()
        frames = []
        for fid, r in renders.items():
            full = cv2.resize(r["img"], (sr.FULL_W, sr.FULL_H), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(ctx.out / "frames" / f"{fid}.png"), full[..., ::-1])
            frames.append({"frame_id": fid, "clip_id": r["sid"][:2], "shot_id": r["sid"], "frame_index": r["i"], "time_s": r["i"] / 24,
                           "file": f"frames/{fid}.png", "sharpness": 100.0, "tenengrad": 1.0, "motion": 0.01,
                           "width": sr.FULL_W, "height": sr.FULL_H})
        write_json(ctx.out / "frames.json", frames)
        return {"keyframes": len(frames)}
    return run


def install_fake_workers(monkeypatch, renders):
    def fake_run_worker(ctx, env, script, job, tag):
        job_dir = ctx.out / f"_job_{tag}"
        job_dir.mkdir(parents=True, exist_ok=True)
        if script == "sam3_worker.py" and job["mode"] == "union":
            (job_dir / "masks").mkdir()
            dets = {}
            for fr in job["frames"]:
                lab = renders[fr["id"]]["label"]
                m = cv2.resize((lab == 5).astype(np.uint8) * 255, (sr.FULL_W, sr.FULL_H), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(job_dir / "masks" / f"{fr['id']}.png"), m)
                dets[fr["id"]] = [{"label": "person", "score": 0.9, "box": [0, 0, 1, 1], "area": int((m > 0).sum())}] if m.any() else []
            (job_dir / "detections.json").write_text(json.dumps(dets))
            return {}
        if script == "sam3_worker.py" and job["mode"] == "instances":
            (job_dir / "inst").mkdir()
            assert job["save_size"] == [RW, RH]
            dets = {}
            name_to_label = {"floor": 0, "wall": 1, "ceiling": 2, "door": 3, "desk": 4}
            for fr in job["frames"]:
                lab = renders[fr["id"]]["label"]
                ms, ds = [], []
                for name, k in name_to_label.items():
                    if name in job["prompts"] and (lab == k).sum() > 50:
                        ms.append(lab == k)
                        ds.append({"label": name, "score": 0.8, "box": [0, 0, 1, 1], "area": int((lab == k).sum())})
                arr = np.stack(ms) if ms else np.zeros((0, RH, RW), bool)
                np.savez_compressed(job_dir / "inst" / f"{fr['id']}.npz", packed=np.packbits(arr, axis=-1), shape=np.array(arr.shape))
                dets[fr["id"]] = ds
            (job_dir / "detections.json").write_text(json.dumps(dets))
            return {}
        if script == "mapanything_worker.py":
            (job_dir / "pred").mkdir()
            (job_dir / "inputs").mkdir()
            frames_res = []
            rng = np.random.default_rng(0)
            for fr in job["frames"]:
                r = renders[fr["id"]]
                cy, cx = CROP["crop_y"], CROP["crop_x"]
                depth = r["depth"][cy:cy + MODEL_H, cx:cx + MODEL_W] * sr.S_MODEL
                depth = depth * (1 + rng.normal(0, 0.002, depth.shape)).astype(np.float32)
                dynm = cv2.imread(fr["mask_path"], cv2.IMREAD_GRAYSCALE)
                dynm = cv2.resize(dynm, (RW, RH), interpolation=cv2.INTER_NEAREST)[cy:cy + MODEL_H, cx:cx + MODEL_W] > 127
                img = r["img"][cy:cy + MODEL_H, cx:cx + MODEL_W].copy()
                img[dynm] = 0
                cv2.imwrite(str(job_dir / "inputs" / f"{fr['id']}.png"), img[..., ::-1])
                np.savez_compressed(job_dir / "pred" / f"{fr['id']}.npz", depth=depth.astype(np.float32),
                                    conf=np.full(depth.shape, 5.0, np.float16), valid=~dynm, dyn=dynm,
                                    K=K_MODEL.astype(np.float32), cam2world=sr.c2w_true_to_model(r["c2w"]).astype(np.float32))
                frames_res.append({"id": fr["id"], "crop": CROP, "metric_scaling_factor": 1.0})
            res = {"model_size": [MODEL_W, MODEL_H], "orig_size": [sr.FULL_W, sr.FULL_H], "weights": job["weights"], "device": "cpu",
                   "precision": "fp32", "fp32_fallback": False, "inference_s": 0.1, "total_s": 0.1, "frames": frames_res}
            (job_dir / "result.json").write_text(json.dumps(res))
            (job_dir / "_worker_stats.json").write_text(json.dumps({"peak_gb": 0.1}))
            return res
        raise AssertionError(f"unexpected worker {script}")

    for mod in (masks, geometry, segment):
        monkeypatch.setattr(mod, "run_worker", fake_run_worker)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory, renders):
    mp = pytest.MonkeyPatch()
    install_fake_workers(mp, renders)
    ws = Workspace(tmp_path_factory.mktemp("work") / "synthetic")
    ws.save_scene({"scene": "synthetic", "clips": []})
    cfg = load_config(overrides={"device": "cpu", "segment": {"use_vlm": False}, "geometry": {"points_per_frame": 20000}})
    specs = [
        StageSpec("s0_ingest", "ingest", [], [], fake_ingest(renders)),
        StageSpec("s0b_masks", "masks", ["s0_ingest"], [], masks.run),
        StageSpec("s1_geometry", "geometry", ["s0_ingest", "s0b_masks"], [], geometry.run),
        StageSpec("s2_segment", "segment", ["s0_ingest", "s1_geometry"], [], segment.run),
    ]
    recs = {s.name: run_stage(s, ws, cfg, assume_yes=True) for s in specs}
    yield ws, recs
    mp.undo()


def test_all_stages_done(pipeline):
    ws, recs = pipeline
    for name, rec in recs.items():
        assert rec["status"] == "done", name
        assert "wall_s" in rec["resources"]
    for s in ("s0b_masks", "s1_geometry", "s2_segment"):
        assert (ws.stage_dir(s) / "report.html").exists()


def test_characters_removed_from_cloud(pipeline):
    ws, _ = pipeline
    d = np.load(ws.stage_dir("s1_geometry") / "points.npz")
    # map model points back to the true frame
    true = (d["xyz"] - sr.T_MODEL) @ sr.R_MODEL / sr.S_MODEL
    lo, hi = sr.CHAR
    inside = np.all((true > lo + 0.02) & (true < hi - 0.02), axis=1)
    assert inside.sum() == 0


def test_metric_scale_and_floor(pipeline):
    ws, recs = pipeline
    frame = read_json(ws.stage_dir("s2_segment") / "scene_frame.json")
    assert frame["scale_source"] == "anchors", frame
    assert frame["up_source"] == "floor_plane"
    assert abs(frame["scale"] * sr.S_MODEL - 1.0) < 0.02, frame["scale"]
    room = read_json(ws.stage_dir("s2_segment") / "room.json")
    size = np.array(room["max"]) - np.array(room["min"])
    dims = sorted(size[:2].tolist())
    assert abs(dims[0] - 3.0) < 0.3 and abs(dims[1] - 4.0) < 0.3, size
    assert abs(size[2] - 2.6) < 0.15, size
    assert room["faces"]["ceiling"]["source"] == "observed"


def test_objects_found_with_sizes(pipeline):
    ws, _ = pipeline
    objs = read_json(ws.stage_dir("s2_segment") / "objects.json")["objects"]
    desks = [o for o in objs if o["label"] == "desk"]
    doors = [o for o in objs if o["label"] == "door"]
    assert len(desks) == 1, [o["id"] for o in objs]
    assert len(doors) == 1
    ds = sorted(desks[0]["bbox"]["size"][:2])
    assert abs(ds[0] - 0.6) < 0.05 and abs(ds[1] - 1.2) < 0.06, desks[0]["bbox"]
    assert abs(desks[0]["bbox"]["size"][2] - 0.75) < 0.04
    assert doors[0]["category"] == "structure"
    assert abs(doors[0]["bbox"]["size"][2] - 2.03) < 0.05
    assert desks[0]["fully_observed"] and doors[0]["fully_observed"]


def test_cameras_rigid_consistency(pipeline):
    ws, _ = pipeline
    cams = read_json(ws.stage_dir("s2_segment") / "cameras_scene.json")["frames"]
    est = np.array([np.array(c["cam2world_opencv"])[:3, 3] for c in cams])
    true = np.array([c2w[:3, 3] for _, c2w in sr.cameras()])
    de = np.linalg.norm(est[:, None] - est[None], axis=-1)
    dt = np.linalg.norm(true[:, None] - true[None], axis=-1)
    assert np.abs(de - dt).max() < 0.08
    # camera heights above the floor are preserved (z-up, floor at 0)
    assert np.abs(est[:, 2] - true[:, 2]).max() < 0.08


def test_cache_invalidation_downstream(pipeline):
    ws, recs = pipeline
    # same config -> keys stable
    rec = ws.load_record("s2_segment")
    assert rec["key"] == recs["s2_segment"]["key"]
