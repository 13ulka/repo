"""MapAnything worker (PyTorch, MPS/CPU). Runs inside envs/geom.

Job JSON:
  frames: [{"id", "path", "mask_path" | null}]
  weights: HF repo id
  device: "mps" | "cpu"
  precision: "auto" | "fp16" | "fp32"
  resolution_long_side, memory_efficient, minibatch_size
  out_dir
Writes:
  inputs/<id>.png                 masked, resized model input (dynamic pixels = black)
  pred/<id>.npz                   depth (H,W) f32, conf (H,W) f16, valid (H,W) bool, dyn (H,W) bool,
                                  K (3,3) model-res intrinsics, cam2world (4,4) OpenCV
  result.json                     per-frame crop params, model size, run info
  _worker_stats.json              peak MPS driver memory
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PATCH = 14


def model_input_size(w0, h0, long_side):
    if w0 >= h0:
        w = (long_side // PATCH) * PATCH
        h = max(PATCH, int(round(h0 * w / w0 / PATCH)) * PATCH)
    else:
        h = (long_side // PATCH) * PATCH
        w = max(PATCH, int(round(w0 * h / h0 / PATCH)) * PATCH)
    return w, h


def resize_crop(img: Image.Image, w: int, h: int, resample):
    w0, h0 = img.size
    s = max(w / w0, h / h0)
    rw, rh = int(round(w0 * s)), int(round(h0 * s))
    ox, oy = (rw - w) // 2, (rh - h) // 2
    out = img.resize((rw, rh), resample).crop((ox, oy, ox + w, oy + h))
    return out, {"scale": s, "resized_w": rw, "resized_h": rh, "crop_x": ox, "crop_y": oy, "w": w, "h": h}


class PeakSampler:
    def __init__(self, device: str):
        self.device = device
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            if self.device == "mps":
                self.peak = max(self.peak, torch.mps.driver_allocated_memory())
            self._stop.wait(0.5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(2)


def run_inference(model, views, job, use_amp: bool, amp_dtype: str):
    with torch.inference_mode():
        return model.infer(
            views,
            memory_efficient_inference=bool(job["memory_efficient"]),
            minibatch_size=(int(job["minibatch_size"]) or None),
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
        )


def has_nan(preds) -> bool:
    return any(not torch.isfinite(p["depth_z"]).all() or not torch.isfinite(p["camera_poses"]).all() for p in preds)


def main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())
    out = Path(job["out_dir"])
    device = job["device"]
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but torch.backends.mps.is_available() is False")
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    t0 = time.time()
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "pred").mkdir(parents=True, exist_ok=True)

    frames = job["frames"]
    w0, h0 = Image.open(frames[0]["path"]).size
    W, H = model_input_size(w0, h0, int(job["resolution_long_side"]))
    crops, dyn_masks, input_paths = {}, {}, []
    for fr in frames:
        img = Image.open(fr["path"]).convert("RGB")
        if img.size != (w0, h0):
            raise SystemExit(f"All frames must share a resolution; {fr['id']} is {img.size}, expected {(w0, h0)}")
        small, crop = resize_crop(img, W, H, Image.LANCZOS)
        arr = np.array(small)
        dyn = np.zeros((H, W), bool)
        if fr.get("mask_path"):
            m, _ = resize_crop(Image.open(fr["mask_path"]).convert("L"), W, H, Image.NEAREST)
            dyn = np.array(m) > 127
            arr[dyn] = 0  # VGGT-family convention: masked pixels set to 0
        p = out / "inputs" / f"{fr['id']}.png"
        Image.fromarray(arr).save(p)
        input_paths.append(str(p))
        crops[fr["id"]] = crop
        dyn_masks[fr["id"]] = dyn
    print(f"prepared {len(frames)} inputs at {W}x{H} in {time.time() - t0:.1f}s", flush=True)

    model = MapAnything.from_pretrained(job["weights"]).to(device)
    model.eval()
    norm_type = getattr(getattr(model, "encoder", None), "data_norm_type", None) or "dinov2"
    views = load_images(input_paths, resize_mode="fixed_size", size=(W, H), norm_type=norm_type, patch_size=PATCH)
    if len(views) != len(frames):
        raise SystemExit(f"load_images returned {len(views)} views for {len(frames)} frames")
    for v in views:
        if tuple(v["img"].shape[-2:]) != (H, W):
            raise SystemExit(f"unexpected view size {tuple(v['img'].shape)}; expected {(H, W)}")

    precision = job["precision"]
    if precision == "auto":
        use_amp, amp_dtype = (device == "mps"), "fp16"
    else:
        use_amp, amp_dtype = precision != "fp32", precision
    fallback = False
    t1 = time.time()
    with PeakSampler(device) as sampler:
        preds = run_inference(model, views, job, use_amp, amp_dtype)
        if use_amp and has_nan(preds):
            print("non-finite outputs in fp16; retrying in fp32", flush=True)
            fallback = True
            del preds
            if device == "mps":
                torch.mps.empty_cache()
            preds = run_inference(model, views, job, False, "fp32")
        if has_nan(preds):
            raise SystemExit("MapAnything produced non-finite outputs even in fp32")
    infer_s = time.time() - t1
    print(f"inference on {len(frames)} views took {infer_s:.1f}s", flush=True)

    per_frame = []
    for fr, p in zip(frames, preds):
        depth = p["depth_z"][0, ..., 0].float().cpu().numpy()
        conf = p["conf"][0].float().cpu().numpy()
        valid = p["mask"][0, ..., 0].bool().cpu().numpy() if p["mask"].ndim == 4 else p["mask"][0].bool().cpu().numpy()
        K = p["intrinsics"][0].float().cpu().numpy()
        c2w = p["camera_poses"][0].float().cpu().numpy()
        dyn = dyn_masks[fr["id"]]
        np.savez_compressed(out / "pred" / f"{fr['id']}.npz", depth=depth, conf=conf.astype(np.float16), valid=valid & ~dyn, dyn=dyn, K=K, cam2world=c2w)
        msf = p.get("metric_scaling_factor")
        per_frame.append({"id": fr["id"], "crop": crops[fr["id"]], "metric_scaling_factor": float(msf.reshape(-1)[0]) if msf is not None else None})

    result = {
        "model_size": [W, H], "orig_size": [w0, h0], "weights": job["weights"], "device": device,
        "precision": "fp32" if (fallback or not use_amp) else amp_dtype, "fp32_fallback": fallback,
        "inference_s": infer_s, "total_s": time.time() - t0, "frames": per_frame,
        "torch": torch.__version__,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "_worker_stats.json").write_text(json.dumps({"peak_gb": sampler.peak / 1024**3}))


if __name__ == "__main__":
    main(sys.argv[1])
