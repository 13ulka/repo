# SceneForge

Reconstructs one film location from short clips into an editable Blender scene.
Current state: **stages 0–2 implemented** (ingest → character masks → geometry → objects/scale) plus a Blender preview.
Stages 3–6 (meshes/textures, inferred completion, verification loop, export) come after the first real run.

Plan and tool choices: [`docs/PLAN.md`](docs/PLAN.md). Tool research archive (licenses, Mac status, alternatives): [`docs/research/`](docs/research/README.md).

## Setup (macOS, Apple Silicon)

Run these in Terminal, from the repo folder:

```zsh
scripts/doctor.sh "/path/clip.mkv"      # read-only check of the machine and clips
scripts/setup_mac.sh                     # 3 envs from lock files + model downloads (asks before each)
source .venv/bin/activate
```

Environments (all Python 3.12, created by `uv` from `locks/*.txt`, resolved for macOS arm64):

| env | contents | why separate |
|---|---|---|
| `.venv` | CLI, OpenCV, PySceneDetect, Open3D | orchestrator |
| `envs/geom` | PyTorch 2.14 (MPS), MapAnything @ `3d10cf7` | MapAnything pins opencv 4.10 |
| `envs/mlx` | MLX 0.32, mlx-vlm 0.7.2 (SAM 3.1) | mlx-vlm needs opencv ≥ 4.12, transformers 5 |
| Blender.app | Blender 5.2 LTS | `bpy` from PyPI requires Python 3.13 |

For stage 2, the VLM runs in LM Studio: `lms server start && lms load qwen/qwen3-vl-8b`.
Without it, stage 2 continues with the default class list and a warning.

## Run

```zsh
sceneforge run "/Users/i300/Movies/IINA Clips/Room RUMI.mkv" "/Users/i300/Movies/IINA Clips/Room RUMI 2.mkv" --scene rumi_room
sceneforge status --scene rumi_room
open work/rumi_room/preview/preview.blend
```

Single stages: `sceneforge ingest|masks|geometry|segment --scene rumi_room [--force] [--device mps|cpu] [--yes]`.
Config overrides: `--config my.toml` (deep-merged over `configs/default.toml`).

Every stage writes to `work/<scene>/<stage>/`:
- `_stage.json`: cache key, time, memory (process tree RSS, system delta, worker MPS/MLX peak), warnings.
- `report.html`: visual checks (contact sheets, masks, projections, objects).
- `log.txt`.

A stage is skipped when its inputs, config and code are unchanged. Use `--force` to recompute.
A failed stage leaves `<stage>.tmp/` for inspection and never replaces the previous good output.

| stage | output |
|---|---|
| `s0_ingest` | shots (PySceneDetect adaptive), adaptive 2–10 fps keyframes, blur/dark filtering |
| `s0b_masks` | SAM 3.1 (MLX) masks of characters, dilated; heavily masked frames excluded |
| `s1_geometry` | MapAnything joint poses + depth for all shots, fused cloud, per-shot registration check |
| `s2_segment` | VLM inventory → SAM 3.1 instances → 3D association → objects; floor/Manhattan alignment; metric scale from fully visible anchors (door, desk, table, bed); room box with observed/inferred faces |
| `preview` | `preview.blend`: colored point cloud, one camera per keyframe with the frame as background, object boxes, room box |

## Tests

```zsh
make test
```

The tests need no models. They cover ingest on synthetic video, and the full masks → geometry → segment chain on an analytic ray-cast room.
In that room the worker outputs are simulated, and the model world is rotated and scaled 0.7× on purpose.
The tests check metric scale (±2%), room size, object sizes, removal of the character from the cloud, and camera rigidity.
