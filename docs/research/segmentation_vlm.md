# Shot detection, segmentation, detection, VLM (checked 2026-09-24)

GitHub API/HF blocked; repos shallow-cloned, PyPI JSON queried. V = verified in repo/code/PyPI, U = search snippets / memory.

## Shot detection
| Name | URL | License | Last update | Mac status | Notes |
|---|---|---|---|---|---|
| PySceneDetect 0.7.1 | github.com/Breakthrough/PySceneDetect | BSD-3 (V) | commit 2026-09-21; PyPI 0.7.1 2026-07-22 (V) | CPU/OpenCV | `scenedetect[opencv]`, Adaptive/Content detectors |
| TransNetV2 | github.com/soCzech/TransNetV2 | MIT (V) | 2021-07-28 (V) | TF; tensorflow-metal on Mac (U) | unmaintained but strong |
| transnetv2-pytorch 1.0.5 | github.com/allenday/transnetv2_pytorch | MIT (V) | PyPI 2025-06-01 (V) | `--device auto/cuda/mps/cpu`, MPS path in code (V) | small |
| ffmpeg scene filter | `select='gt(scene,0.3)'` | LGPL/GPL | — | CPU | quick pre-pass; fades/flashes cause false cuts |

## Segmentation / tracking
| Name | URL | License | Last update | Mac status | Notes |
|---|---|---|---|---|---|
| SAM 2 / 2.1 | github.com/facebookresearch/sam2 | Apache-2.0 + BSD-3 (V) | 2024-12-15; PyPI 1.1.0 (V) | notebooks: "Support for MPS devices is preliminary… degraded" (V); `build_sam.py` defaults cuda; #687 "Placeholder storage… MPS" crash on M4 (V) | 38.9M–224M params |
| SAM 3 / 3.1 official | github.com/facebookresearch/sam3 | SAM License 2025-11-19: commercial OK, attribution, no reverse engineering, trade controls (V); HF weights gated | commit 2026-09-18; 3.1 Object Multiplex 2026-03-27 (V) | **no Mac**: README requires CUDA ≥12.6; Triton kernels in import path (V); #164 asks for CPU/MPS | 848M; text prompts, video tracking, detector |
| sam3-mac | github.com/benreichman/sam3-mac | MIT (V) | 2026-05-18 (V) | runtime patches for MPS/CPU (V) | image-only (U) |
| **SAM 3 / 3.1 in mlx-vlm** | github.com/Blaizzy/mlx-vlm `mlx_vlm/models/sam3`, `sam3_1` | MIT code; SAM License weights | mlx-vlm 0.7.2 on 2026-09-21, commit 2026-09-23 (SHA 5e1aa686) (V) | **native MLX** (V) | `Sam3Predictor.predict(image, text_prompt, boxes=None)` → boxes/masks/scores at image size; `Sam3VideoPredictor`; M3 Max bf16: ~1.0 s per detection at 1008px, tracker ~0.2 s/frame for ≤16 objects (README). Models `mlx-community/sam3.1-bf16`, `facebook/sam3`. Also `sam3d_objects`, `sam3d_body` ports present |
| EfficientTAM | github.com/yformer/EfficientTAM | Apache-2.0 (V) | 2025-01-05, "Mac MPS backend support" (V) | MPS/CPU | point/box prompts only |
| SAM2 MLX ports | eisneim/sam2.1_mlx, avbiswas/sam2-mlx, PyPI mlx-sam 0.3.0 (Py≥3.14) | U | — | MLX | community |

## Open-vocab detection
| Name | License | Last update | Mac | Notes |
|---|---|---|---|---|
| Grounding DINO | Apache-2.0 (V) | 2024-08-12 | no MPS code (V); pure-torch fallback (U) | stale; use transformers |
| Grounded-SAM-2 | Apache-2.0 (V) | 2025-11-11 | no MPS handling (V) | CUDA glue |
| Florence-2 | MIT (U) | — | MLX port in mlx-vlm (V) | `<OD>`, phrase grounding; 0.23B/0.77B |
| OWLv2 | Apache-2.0 (U) | — | transformers, MPS likely (U) | slower |
| YOLO-World | GPL-3.0 (V); Ultralytics AGPL-3.0 (V) | 2025-02-27 | Ultralytics MPS; SAM3 pin_memory bug #22954 (U) | license issue |
| SAM 3 detector | SAM License | — | via MLX (V) | replaces DINO+SAM |

## Local VLMs
| Model | License | Sizes | MLX | Grounding | Notes |
|---|---|---|---|---|---|
| Qwen3-VL | Apache-2.0 (V) | 2B/4B/8B/32B + MoE (U) | mlx-vlm `qwen3_vl(_moe)` (V); LM Studio MLX+GGUF (U) | JSON boxes 0–1000 (U); llama.cpp non-square bug #16880 (U) | repo 2026-01-30 (V); 8B 4-bit ≈5–6 GB (U) |
| Qwen3.5 | Apache-2.0 (U) | 0.8B–27B + MoE (U) | `qwen3_5` (V) | probably (U) | newer; benchmark |
| Gemma 4 | Apache-2.0 (U) | E2B/E4B/12B?/26B-A4B/31B (U) | `gemma4` (V) | `[ymin,xmin,ymax,xmax]` 0–1000 (U) | alternative |
| InternVL3/3.5 | MIT code (V) | 1B–78B | `internvl_chat` (V) | `<box>` | 2025-09-22 |
| Molmo2 | Apache-2.0 (V) | 4B/8B/7B-O (U) | `molmo2` (V) | points, video pointing (U) | 2026-03-18 |

Serving: `mlx_vlm.server` (OpenAI-style) or LM Studio `/v1/chat/completions` with images (U). LM Studio mlx-engine #325 (May 2026) image-token mismatch on Qwen3.6 (U).

## Characters in animation
No published SAM 3 evaluation on cartoons (unverified). Test prompts "person", "cartoon character", "animal", "creature"; accept masks stable across frames. Sapiens2 (mlx-vlm port) trained on real humans — likely weaker on stylized characters (U). Fallback: VLM names characters → SAM 3 text/box prompts.

## Recommendation (adopted)
PySceneDetect adaptive (+ transnetv2-pytorch optional); Laplacian/Tenengrad sharpness; SAM 3.1 via mlx-vlm for detection+masks; Qwen3-VL-8B 4-bit via LM Studio (alt Gemma 4). Avoid YOLO-World/Ultralytics (GPL/AGPL) and official SAM 3 on Mac.
