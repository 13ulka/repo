# Geometry: pose-free reconstruction, mono depth, SfM (checked 2026-09-24)

Verified by shallow clone (LICENSE, README, requirements, grep for mps/cuda/xformers/flash/curope, git log) + PyPI + homebrew-core. HF model cards and GitHub issues were NOT directly readable (proxy); weight licenses come from repo READMEs. No model was run on a Mac.

## Multi-view feed-forward (pose-free)
| name | repo | code lic. | weights lic. | last commit | Mac/MPS status (evidence) | memory / limits | notes |
|---|---|---|---|---|---|---|---|
| VGGT | github.com/facebookresearch/vggt | VGGT License v1 (commercial OK, no military) | VGGT-1B NC; VGGT-1B-Commercial commercial (gated) | 2026-05-18 | Likely OK: commit 2025-04-24 "use float32 in pos embed for MPS"; xformers optional (SDPA fallback); deps torch 2.3.1, einops, safetensors. README autocast snippet CUDA-only | May-2026 fix: ~2-3x more frames per memory | unordered input; README: mask unwanted pixels by setting to 0/1 |
| VGGT-Ω | github.com/facebookresearch/vggt-omega | FAIR Noncommercial | gated NC | 2026-09-22 | Partial: MPS fp32 branch + SDPA, but `torch.autocast(device_type="cuda")` hard-coded → patch | A100: 100 fr 13.4 GB, 200 fr 20.8 GB, 500 fr 43 GB @624×416 | CVPR'26; reportedly handles dynamic scenes (U) |
| VGGT-Long | github.com/DengKaiCQ/VGGT-Long | as VGGT | VGGT | 2026-02-09 | CUDA-only as written (cuda capability call, xformers 0.0.28, faiss-gpu) | chunked, km-scale | overkill for a room |
| MASt3R / MASt3R-SfM | github.com/naver/mast3r | CC BY-NC-SA 4.0 | NC | 2025-06-30 | `--device mps` + `PYTORCH_ENABLE_MPS_FALLBACK=1` (dust3r #31 on M2); curope optional; ASMK Cython build | grows with #pairs | retrieval + sparse GA for unordered sets |
| DUSt3R | github.com/naver/dust3r | CC BY-NC-SA | NC | 2025-07-01 | as MASt3R | ~50 images | superseded |
| π³ / Pi3X | github.com/yyfz/Pi3 | BSD | CC BY-NC 4.0 | 2026-05-18 | Probably OK: xformers optional, fp32 SDPA fallback, curope optional | n/a | permutation-equivariant; Pi3X takes pose/intrinsics/depth conditioning |
| **MapAnything** | github.com/facebookresearch/map-anything | Apache 2.0 | `map-anything` CC-BY-NC; `map-anything-apache` Apache | 2026-08-07 (SHA 3d10cf7a) | **Official MPS**: commit 2026-03-22 "Add MPS inference support (#131)"; `utils/device.py` CUDA>MPS>CPU, bf16 disabled on MPS → fp16 autocast | memory_efficient_inference: "up to 2000 views on 140 GB" | unordered; optional intrinsics/pose/depth; wraps VGGT, VGGT-Ω, Pi3, MASt3R, DA3, MoGe. NB: pins opencv-python-headless==4.10.0.84, uniception==0.1.7; `[colmap]` extra pins pycolmap==3.10.0 |
| Depth Anything 3 | github.com/ByteDance-Seed/Depth-Anything-3 | Apache 2.0 | GIANT/LARGE/NESTED CC BY-NC; BASE/SMALL/METRIC-L/MONO-L Apache | 2026-07-27; PyPI 0.1.1 (2026-03) | xformers is hard dep (no macOS wheel) → `--no-deps`, falls back to torch SwiGLU; README hard-codes cuda; community Mac ports exist | DA3-Streaming <12 GB | any-view pose+depth; ~VGGT-level per own README |
| CUT3R | github.com/CUT3R/CUT3R | CC BY-NC-SA | NC | 2025-08-27 | unverified | linear in frames | recurrent; dynamic training data |
| Fast3R | github.com/facebookresearch/fast3r | FAIR NC | NC | 2025-05-07 | FAQ covers no-FlashAttention; unverified MPS | 1000+ images | inactive |
| StreamVGGT | github.com/wzzheng/StreamVGGT | CC BY-NC-SA | NC | 2025-10-22 | FlashAttention-2 | streaming | ordered only |
| VGG-T³ | github.com/nv-dvl/vgg-ttt | NVIDIA OneWay NC | NC | 2026-05-25 | `.cuda()` in README | 1k images <1 min | CVPR'26 |

Dynamic-aware: MonST3R (CC BY-NC-SA, 2025-06-16, ~23 GB for 65 frames, predicts dynamic masks). VGGT4D / PAGE-4D found in search, not checked. Practical route: segment characters and zero pixels before inference.

## Monocular depth
| name | repo | code | weights | last commit | Mac/MPS | notes |
|---|---|---|---|---|---|---|
| Depth Anything V2 | github.com/DepthAnything/Depth-Anything-V2 | Apache | Small Apache; B/L/G CC-BY-NC | 2026-03-24 | README selects `mps`; CoreML version exists | relative disparity |
| DA3 Mono/Metric | DA3 repo | Apache | MONO-L, METRIC-L Apache | 2026-07-27 | xformers workaround | best permissive mono |
| MoGe-2 | github.com/microsoft/MoGe | MIT | not verified | 2026-08-19 | main: "macOS is not supported" (MoGe-3 needs FlexGEMM/Triton, CUDA 13 wheels). Pin `0744441` (2025-11-02) for MoGe-2 | metric point map + FOV |
| Depth Pro | github.com/apple/ml-depth-pro | Apple sample-code licence | same | 2026-09-11 | CLI auto-selects mps | metric depth + focal |
| UniDepthV2 | github.com/lpiccinelli-eth/UniDepth | CC BY-NC 4.0 | NC | 2025-05-18 | pins xformers, CUDA 11.8+ | poor fit |

## Classic SfM
| name | status |
|---|---|
| COLMAP | BSD; 4.2.0 tag 2026-08-31; homebrew-core `colmap` 4.2.0; pycolmap 4.2.0 macOS arm64 wheels cp310–cp313; CPU SIFT on Mac |
| GLOMAP | deprecated, merged into COLMAP as global mapper (`colmap global_mapper`); no brew formula |
| hloc | Apache; 2025-12-10; SuperPoint/SuperGlue weights believed NC (Magic Leap) — prefer LightGlue+ALIKED/DISK |

## Recommendation (adopted)
1. Primary: MapAnything (fp16 autocast on MPS), weights per license profile.
2. Comparison via the same wrapper: VGGT-1B(-Commercial), Pi3X, VGGT-Ω (needs autocast patch).
3. Characters: segment and zero pixels before inference, confidence filtering after.
4. Fallback: COLMAP 4.2 global mapper with character masks; MASt3R-SfM next.
5. Mono depth: Depth Pro (MPS) or DA3-Mono/Metric-L (`--no-deps`). Avoid MoGe main on Mac.
