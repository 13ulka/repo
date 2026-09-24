# Tool research archive

Snapshot date: **2026-09-24**. Sources were checked by shallow-cloning repos (LICENSE, README, code, git log) and PyPI JSON.
Legend: **V** = verified in primary source, **U** = secondary source / memory / inference. Nothing here was benchmarked on the target Mac.

Files:
- `geometry.md` — pose-free multi-view reconstruction, monocular depth, SfM.
- `segmentation_vlm.md` — shot detection, SAM family, open-vocab detection, local VLMs.
- `splats_image_to_3d.md` — Gaussian splatting on Mac, splat→mesh, Blender splats, image-to-3D, bpy.
- `generation_verification.md` — inpainting/editing, novel-view models, metrics, differentiable rendering, lighting.

## License profiles

SceneForge has a `license_profile` setting (`configs/default.toml`). Default is `noncommercial`.
To rebuild the toolset for commercial use, swap these components:

| Component | `noncommercial` (default) | `commercial` replacement | Notes |
|---|---|---|---|
| MapAnything weights | `facebook/map-anything` (CC-BY-NC) | `facebook/map-anything-apache` (Apache-2.0) | same code, wired in config |
| Alt. geometry | VGGT-Ω (FAIR NC), π³/Pi3X (CC BY-NC), VGGT-1B (NC) | VGGT-1B-Commercial (gated), DA3 Base/Small/Metric-L (Apache) | |
| SAM 3.1 | SAM License | same — commercial use allowed with attribution + license copy | |
| VLM | Qwen3-VL (Apache) | same; or Gemma 4 (Apache) | |
| Image-to-3D | TRELLIS.2 (MIT) + RMBG-2.0 (CC BY-NC) | TRELLIS.2 with SAM-mask background removal instead of RMBG | Hunyuan3D: territory-restricted (EU/UK/KR excluded) |
| Inpainting | FLUX.1 Fill/Kontext dev (NC) | FLUX.2-klein-4B (Apache), Qwen-Image-Edit (Apache) | |
| Splats | Brush (Apache) | same; OpenSplat is AGPL | |
| Mesh | Open3D (MIT), PyMeshLab (GPL-3) | same (GPL only matters if redistributing) | |
| Metrics | DINOv2 (Apache) | same; DINOv3 has custom license | |
| Diff. render | Mitsuba 3 (BSD) | same | nvdiffrast is NC |
