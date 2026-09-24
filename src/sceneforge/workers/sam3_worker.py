"""SAM 3.1 (MLX, mlx-vlm) worker. Runs inside envs/mlx.

Job JSON:
  frames: [{"id": str, "path": str}]
  prompts: [str]
  score_threshold: float
  model: HF repo id (e.g. mlx-community/sam3.1-bf16)
  mode: "union"      -> masks/<id>.png (255 = any prompt matched) + detections.json
        "instances"  -> inst/<id>.npz (packed boolean masks at save_width) + detections.json
  save_size: [W, H] (instances mode) — masks are resized to exactly this size
  out_dir: str
Writes result.json and _worker_stats.json (peak MLX memory).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image


def load_predictor(model_id: str, threshold: float):
    from mlx_vlm.models.sam3.generate import Sam3Predictor
    from mlx_vlm.utils import get_model_path, load_model

    mp = get_model_path(model_id)
    cfg = json.loads((Path(mp) / "config.json").read_text())
    model_type = str(cfg.get("model_type", ""))
    is_31 = any(tag in model_type for tag in ("sam3.1", "sam3_1")) or any(tag in model_id.lower() for tag in ("sam3.1", "sam3_1", "sam3-1"))
    if is_31:
        from mlx_vlm.models.sam3_1.generate import predict_multi
        from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor as Proc
    else:
        from mlx_vlm.models.sam3.generate import predict_multi
        from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor as Proc
    model = load_model(mp)
    processor = Proc.from_pretrained(str(mp))
    return Sam3Predictor(model, processor, score_threshold=threshold), predict_multi, model_type


def peak_gb() -> float:
    try:
        return mx.get_peak_memory() / 1024**3
    except AttributeError:
        return mx.metal.get_peak_memory() / 1024**3


def main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())
    out = Path(job["out_dir"])
    mode = job["mode"]
    t0 = time.time()
    predictor, predict_multi, model_type = load_predictor(job["model"], job["score_threshold"])
    print(f"loaded {job['model']} ({model_type}) in {time.time() - t0:.1f}s", flush=True)

    (out / ("masks" if mode == "union" else "inst")).mkdir(parents=True, exist_ok=True)
    detections = {}
    n = len(job["frames"])
    t1 = time.time()
    for k, fr in enumerate(job["frames"]):
        img = Image.open(fr["path"]).convert("RGB")
        W, H = img.size
        res = predict_multi(predictor, img, job["prompts"], score_threshold=job["score_threshold"])
        labels = list(res.labels or [])
        masks = np.asarray(res.masks)
        if masks.ndim == 3 and masks.shape[0] > 0 and masks.shape[1:] != (H, W):
            masks = np.stack([np.array(Image.fromarray((m > 0).astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 0 for m in masks])
        masks = masks.astype(bool) if masks.size else np.zeros((0, H, W), bool)
        dets = [
            {"label": labels[i], "score": float(res.scores[i]), "box": [float(v) for v in res.boxes[i]], "area": int(masks[i].sum())}
            for i in range(len(labels))
        ]
        detections[fr["id"]] = dets
        if mode == "union":
            union = masks.any(axis=0) if len(masks) else np.zeros((H, W), bool)
            Image.fromarray((union * 255).astype(np.uint8)).save(out / "masks" / f"{fr['id']}.png")
        else:
            sw, sh = (int(v) for v in job["save_size"])
            small = np.stack([np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((sw, sh), Image.NEAREST)) > 127 for m in masks]) if len(masks) else np.zeros((0, sh, sw), bool)
            np.savez_compressed(out / "inst" / f"{fr['id']}.npz", packed=np.packbits(small, axis=-1), shape=np.array(small.shape))
        if (k + 1) % 10 == 0 or k + 1 == n:
            rate = (time.time() - t1) / (k + 1)
            print(f"{k + 1}/{n} frames, {rate:.2f}s/frame, ETA {rate * (n - k - 1) / 60:.1f} min, peak {peak_gb():.1f} GB", flush=True)

    (out / "detections.json").write_text(json.dumps(detections))
    (out / "result.json").write_text(json.dumps({"frames": n, "model_type": model_type, "seconds": time.time() - t0}))
    (out / "_worker_stats.json").write_text(json.dumps({"peak_gb": peak_gb()}))


if __name__ == "__main__":
    main(sys.argv[1])
