# Generative completion, metrics, differentiable rendering, lighting (checked 2026-09-24)

V = primary (repo file / PyPI), U = secondary/memory.

## Inpainting / editing
| Name | URL | License | Last update | Mac status | Memory | Notes |
|---|---|---|---|---|---|---|
| ComfyUI | github.com/comfyanonymous/ComfyUI | GPL-3.0 (V) | 2026-09-23 (V) | Apple silicon install section, MPS (V); FP8 not on MPS (U) | — | headless API in server.py: POST /prompt, GET /history/{id}, GET /view, GET /ws, POST /upload/image, /upload/mask, /queue, /interrupt, /free, /object_info, /api/jobs (V) |
| mflux | github.com/filipstrand/mflux | MIT (V) | v0.20.0 2026-09-21 (V) | MLX native (V) | FLUX.2-klein-4B ~15 GB, 9B ~32 GB; Qwen-Image ~58 GB bf16 (V) | FLUX.1 Fill (in/outpaint), Kontext, Redux, depth, ControlNet; FLUX.2 klein edit; Qwen-Image-Edit 2509/2511; Qwen Image 2.1; Z-Image, FIBO, SeedVR2, Depth Pro. Qwen ≤6-bit loses quality (V) |
| FLUX.1 Fill/Kontext dev | github.com/black-forest-labs/flux | code Apache; weights FLUX.1-dev NC (U) | 2025-07-31 (V) | mflux / ComfyUI | 12B | best masked in/outpaint |
| FLUX.2 | github.com/black-forest-labs/flux2 | klein 4B Apache; klein 9B & dev 32B NC (V) | 2026-03-12 (V) | klein in mflux (V) | 4B ~15 GB | no dedicated Fill model found |
| Qwen-Image-Edit 2509/2511 | github.com/QwenLM/Qwen-Image | Apache-2.0 (V repo; weights U) | 2026-02-10 (V) | mflux, ComfyUI native (V) | 20B, ~58 GB bf16 | Qwen-Image-2.0 weights unclear (U) |
| SDXL inpainting 0.1 | HF diffusers | OpenRAIL++ (U) | old | diffusers MPS (U) | ~7 GB | weak baseline; diffusers 0.40.0 (2026-08-20) (V) |

## Novel-view / 3D inpainting / world models
| Name | License | Last update | Mac | Notes |
|---|---|---|---|---|
| SEVA | Stability NC, outputs NC (V) | 2025-06-05 | CUDA/H100 only (V) | not practical |
| ViewCrafter | Apache-2.0 (V) | 2025-12-13 | CUDA/Docker (V), 13.8–23.5 GB A100 | not practical |
| MVInpainter | Apache-2.0 (V) | 2024-11-09 | CUDA | stale |
| InFusion | MIT (V) | 2024-07-15 | CUDA | stale |
| HY-World 2.0 (WorldMirror 2, HY-Pano 2, WorldStereo 2) | Tencent community, excl. EU/UK/KR (V) | 2026-08-13 | CUDA 12.8 recommended (V) | WorldMirror 2 ~1.2B maybe MPS w/ patches (U); HY-Pano-2-Qwen = ~425M LoRA on Qwen-Image (V) → 360° outpaint possible on Mac (U) |
| HunyuanWorld 1.0 / WorldMirror 1 | Tencent (V) | 2026-04/05 | CUDA | |
| Marble (World Labs) | proprietary cloud | active | API | ~$1.20/world (U) |

→ Mac approach: render missing views in Blender → 2D inpaint (FLUX Fill / Qwen Edit) with reprojection for consistency; or remote CUDA for SEVA/ViewCrafter.

## Metrics
| Name | License | Last update | Mac |
|---|---|---|---|
| lpips | BSD (V) | 0.1.4 2021 | torch convnets (U) |
| torchmetrics | Apache-2.0 (V) | 1.9.0 2026-03-09 (V) | pure torch (U) |
| open_clip | MIT (V) | 3.3.0 2026-02-27 (V) | MPS ok (U) |
| DINOv2 | Apache-2.0 (V) | 2026-06-03 | torch.hub/transformers (U) |
| DINOv3 | custom DINOv3 License (V) | 2026-07-15 | gated HF (U) |

## Differentiable rendering / optimization
| Name | License | Last update | Mac |
|---|---|---|---|
| Mitsuba 3 | BSD (V) | 3.9.1 2026-08-07 (V) | arm64 wheels; 3.9.0 adds `metal_*`/`metal_ad_*` variants with HW ray tracing (V); `llvm_ad_*` CPU |
| Dr.Jit | BSD (V) | 1.5.0 2026-08-07 (V) | arm64, Metal backend (V) |
| nvdiffrast | NVIDIA NC (V) | 2025-12-05 | CUDA/OpenGL only (U) |
| PyTorch3D | BSD (V) | repo 2026-09-15; PyPI 0.7.4 (2023) (V) | source build, CPU kernels only (U) |
| Kaolin | Apache-2.0 (V) | 2026-08-14 | macOS CPU-only, many ops missing (V) |
| cmaes | MIT | 0.13.1 2026-08-21 (V) | pure numpy |
| nevergrad | MIT (V) | 1.0.12 2025-04 | pure python |

## Lighting estimation
| Name | License | Last update | Mac |
|---|---|---|---|
| DiffusionLight | MIT (V) | 2024-12-18 | conda/CUDA; SDXL-based, portable (U) |
| DiffusionLight-Turbo / ComfyUI node | ? | 2025-07 (V) | node may run on MPS (U) |
| StyleLight | MIT (V) | 2023-10-09 | CUDA 10.2 / torch 1.7 — dead |

## Recommendation (adopted)
mflux (FLUX.1 Fill; Qwen-Image-Edit-2511 / FLUX.2-klein-4B) + ComfyUI headless adapter; torchmetrics LPIPS/SSIM + DINOv2; Mitsuba 3 `metal_ad_rgb` for gradient refinement, `cmaes` for black-box Blender loop; lighting fitted inside optimization instead of a dedicated estimator.
