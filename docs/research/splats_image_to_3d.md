# Splatting, meshing, Blender splats, image-to-3D, bpy (checked 2026-09-24)

V = primary source (repo/README/PyPI), U = secondary/inferred.

## Gaussian splatting training on Mac
| Name | URL | License | Last update | Mac status | Memory | Notes |
|---|---|---|---|---|---|---|
| OpenSplat | github.com/WebODM/OpenSplat (moved from pierotofy) | AGPLv3 (V) | v1.2.2 Sep 16 (V) | MPS default on macOS, CPU fallback; "MPS support with fused kernels" (V); Mac binaries | ~2 GB / 1M gaussians (V) | COLMAP/nerfstudio/OpenSfM/ODX/OpenMVG in; .ply/.splat/.spz out |
| Brush | github.com/ArthurBrussee/brush | Apache-2.0 (V) | v0.3.0 Sep 14; pushed 2026-09-20 (V) | WebGPU/wgpu → Metal; aarch64-apple-darwin binary (V) | n/a | COLMAP/nerfstudio + masks; .ply |
| msplat | github.com/rayanht/msplat | Apache-2.0 (V) | v1.1.4 2026-08-25 (V) | Metal-native, macOS 14+, Py≥3.12; M4 Max "room" 7K iters 74 s (V) | n/a | `pip install msplat[cli]`; `render_from_pose()` RGB; export_ply; young |
| gsplat-mlx | github.com/RobotFlow-Labs/gsplat-mlx | Apache-2.0 (V) | Sep 2026 | MLX port; renders depth; 2DGS module (V) | n/a | source install; 21★ |
| gsplat | github.com/nerfstudio-project/gsplat | Apache-2.0 (V) | 2026-09-19 | CUDA only; MPS issue #163 open since 2024 (V) | — | splatfacto fails on Apple Silicon (#3290) |
| LichtFeld Studio | github.com/MrNeRF/LichtFeld-Studio | GPL-3.0 (V) | 2026-09-24 | NVIDIA only, CUDA 12.8+ (V) | — | |
| Apple SHARP | github.com/apple/ml-sharp | code OSS; weights research-only (U) | 2025 | single image → 3DGS | — | not for multi-view |

## Splat → mesh
| Name | License | Last push | Mac? | Notes |
|---|---|---|---|---|
| 2DGS | Inria NC (V) | 2026-08-25 | No (CUDA surfel rasterizer) (U) | method = render depth + TSDF, reproducible on Mac |
| PGSR, GOF, SuGaR | NOASSERTION (V) | 2024 | No (U) | stale |
| MILo | NOASSERTION (V) | 2026-04 | No (U) | best quality, NVIDIA only |
| NKSR | NVIDIA (V) | 2026-08 | No, CUDA (V) | |
| **Open3D 0.20.0** | MIT (V) | PyPI 2026-09-16 (V) | arm64 wheels cp310–cp314 (V) | ScalableTSDFVolume, Poisson, BPA |
| **PyMeshLab 2025.7.post1** | GPL-3.0 (V) | 2026-01-30 (V) | arm64 wheels cp310–cp314 (V) | screened Poisson, cleanup, decimation |

## Splats in Blender
| Name | License | Last update | Blender | Notes |
|---|---|---|---|---|
| KIRI 3DGS Render | GPL-2.0 (V) | pushed 2026-09-20, v5.1.0 (V) | 5.1, 5.2 (V) | Mac path fixes in 4.1.5 (U) |
| ReshotAI addon | no license (V) | 2024-08 | old | avoid |
| Blender native | GPL | 5.3 alpha (U) | PLY/SPZ import as "3D Gaussian Splats" PointCloud; Workbench/EEVEE/Cycles; no export | release expected 2026-11-10 (U) |

## Image-to-3D
| Name | License | Last push | Mac status | Memory | Texturing on Mac |
|---|---|---|---|---|---|
| **TRELLIS.2** | MIT code+weights (V); deps: DINOv3 custom, RMBG-2.0 CC BY-NC (V) | 2026-07-10 | official Linux/NVIDIA; ports: shivampkumar/trellis-mac (MPS, 24 GB+, ~5 min/asset, setup.sh) (V); lyonsno/trellis2mlx (MLX, 3–5 GB, ~21 min on 16 GB M2 Pro, preview) (V); pedronaugusto/trellis2-apple, apetersson/TRELLIS.2-apple-silicon (U) | 24 GB NVIDIA official | trellis-mac: PBR bake with Metal `mtldiffrast`, xatlas fallback, hole filling disabled (V) |
| TRELLIS v1 | MIT (V) | 2026-06-26 | CUDA only (V) | 16 GB | superseded |
| Hunyuan3D-2 / mini / mv / Turbo | Tencent Community License, NOT in EU/UK/South Korea (V) | 2025-10-28 | README "supports Macos" (V); shape on MPS; Mac forks (U) | shape 6 GB, +texture 16 GB (V) | texture needs CUDA custom_rasterizer (U) |
| Hunyuan3D-2.1 | same (V) | 2025-10-17 | README claims macOS (V); fork Brainkeys/Hunyuan3D-2.1-mac (U) | 10 + 21 GB (V) | PBR CUDA-bound (U) |
| Hunyuan3D-Omni | same (U) | 2025-10-17 | unchecked | | shape control only |
| ComfyUI-Hunyuan3DWrapper | NOASSERTION (V) | 2026-03-16 | undocumented; texture CUDA (U) | | ComfyUI native Hunyuan3D 2.0/2mv shape only (U) |
| ComfyUI-3D-Pack | MIT (V) | 2025-12-29 | torch 2.5.1+cu124 pinned → no Mac (V) | | |
| Stable Fast 3D | Stability Community (V) | 2025-01 | MPS experimental, Metal texture baker, `PYTORCH_ENABLE_MPS_FALLBACK=1` (V) | ~6 GB | yes (V) |
| SPAR3D | Stability Community (V) | 2025-05 | MPS experimental, macOS 15.2+, tested M4 Max 36 GB (V) | 10.5 / ~7 GB low-VRAM (V) | yes, Metal baker (V) |
| TripoSR | MIT (V) | 2026-06 | pure torch, likely MPS (U) | ~6 GB | vertex colors only |
| TripoSG | MIT (V) | 2025-04 | README CUDA 8 GB (V); Mac claim in blog (U) | 8 GB | shape only |
| Step1X-3D | Apache-2.0 (V) | 2025-09 | No: cu124 + kaolin (V) | | |
| SAM 3D Objects | SAM License (V) | 2025-11 / 2026-06 (V) | no Mac mention officially; **MLX port exists in mlx-vlm `sam3d_objects`** (V, directory present; not evaluated) | | gaussians + textured shapes |

## Blender / bpy
- Blender 5.2 LTS released 2026-07-14, supported to July 2028 (U: blender.org blocked); 5.2.2 current (V, PyPI). 5.3 alpha, release planned 2026-11-10 (U).
- `bpy` 5.2.2 on PyPI (2026-09-15): macosx_11_0_arm64 wheel, **Requires-Python ==3.13.*** (V). → SceneForge drives Blender.app headless instead.

## Recommendation (adopted)
Brush (Apache) or OpenSplat as trainer; msplat / gsplat-mlx for Python renders; TSDF (Open3D) + PyMeshLab for mesh; KIRI addon for splats in 5.2, native in 5.3; TRELLIS.2 via trellis-mac, SF3D/SPAR3D as fast fallback; Hunyuan3D shape-only and license-restricted.
