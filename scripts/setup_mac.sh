#!/usr/bin/env zsh
# SceneForge setup for macOS on Apple Silicon.
#
# Creates three isolated Python 3.12 environments from pinned lock files (resolved for macOS arm64):
#   .venv       orchestrator + CLI (sceneforge)          locks/main.txt
#   envs/geom   PyTorch (MPS) + MapAnything               locks/geom.txt
#   envs/mlx    MLX + mlx-vlm (SAM 3.1)                   locks/mlx.txt
# Then optionally downloads model weights (shows sizes and asks first).
#
# Safe: never deletes anything; existing envs are reused (`uv pip sync` makes them match the lock).
# Usage:  scripts/setup_mac.sh            # envs + interactive model downloads
#         scripts/setup_mac.sh --no-models

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
WITH_MODELS=1
[[ "${1:-}" == "--no-models" ]] && WITH_MODELS=0

say()  { print -P "%F{cyan}==>%f $*"; }
warn() { print -P "%F{yellow}!! %f $*"; }
die()  { print -P "%F{red}xx %f $*"; exit 1; }
confirm() { local ans; read "ans?$1 [y/N] "; [[ "$ans" == [yY]* ]]; }

# ---------------------------------------------------------------- checks
[[ "$(uname -s)" == "Darwin" ]] || die "macOS only"
[[ "$(uname -m)" == "arm64" ]] || die "Run from a native arm64 shell (not Rosetta)."
command -v uv >/dev/null || die "uv not found: brew install uv"
command -v ffmpeg >/dev/null || warn "ffmpeg not found (brew install ffmpeg) — used for clip metadata only"
BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
[[ -x "$BLENDER" ]] && say "Blender: $("$BLENDER" --version 2>/dev/null | head -n1)" || warn "Blender not found at $BLENDER (needed for preview/export)"
FREE_GB=$(df -g "$ROOT" | awk 'NR==2 {print $4}')
say "free disk: ${FREE_GB} GB (envs ~6 GB, models ~15 GB)"
(( FREE_GB > 30 )) || warn "less than 30 GB free"

# ---------------------------------------------------------------- envs
make_env() {  # name dir lockfile
  local name=$1 dir=$2 lock=$3
  if [[ ! -x "$dir/bin/python" ]]; then
    say "creating $name env at $dir (Python 3.12)"
    uv venv --python 3.12 "$dir"
  else
    say "reusing $name env at $dir"
  fi
  say "syncing $name env with $lock"
  uv pip sync --python "$dir/bin/python" "$lock"
}

make_env main .venv locks/main.txt
uv pip install --python .venv/bin/python --no-deps -e .
make_env geom envs/geom locks/geom.txt
make_env mlx envs/mlx locks/mlx.txt

# ---------------------------------------------------------------- smoke tests
say "checking torch/MPS and MapAnything"
envs/geom/bin/python - <<'PY'
import torch, mapanything
print(f"torch {torch.__version__}  mps available={torch.backends.mps.is_available()}  built={torch.backends.mps.is_built()}")
assert torch.backends.mps.is_available(), "MPS not available"
x = torch.randn(512, 512, device="mps")
print("mps matmul ok:", float((x @ x).sum().cpu()) == float((x @ x).sum().cpu()))
PY
say "checking MLX and mlx-vlm SAM 3.1"
envs/mlx/bin/python - <<'PY'
import mlx.core as mx, mlx_vlm
from mlx_vlm.models.sam3_1.generate import predict_multi  # noqa: F401
print("mlx", mx.__version__, "default device:", mx.default_device(), "mlx-vlm", mlx_vlm.__version__)
PY
say "checking orchestrator"
.venv/bin/sceneforge --help >/dev/null && say "sceneforge CLI ok"

# ---------------------------------------------------------------- models
if (( WITH_MODELS )); then
  PROFILE=$(.venv/bin/python -c "from sceneforge.core.config import load_config as l; print(l()['license_profile'])")
  if [[ "$PROFILE" == "commercial" ]]; then MA_REPO=facebook/map-anything-apache; else MA_REPO=facebook/map-anything; fi
  SAM_REPO=$(.venv/bin/python -c "from sceneforge.core.config import load_config as l; print(l()['masks']['model'])")
  for spec in "geom:$MA_REPO" "mlx:$SAM_REPO"; do
    env=${spec%%:*}; repo=${spec#*:}
    size=$(envs/$env/bin/python - "$repo" <<'PY'
import sys
from huggingface_hub import HfApi
try:
    info = HfApi().model_info(sys.argv[1], files_metadata=True)
    print(f"{sum((s.size or 0) for s in info.siblings) / 1e9:.1f} GB")
except Exception as e:
    print(f"unknown ({type(e).__name__})")
PY
)
    if confirm "Download $repo ($size) into the Hugging Face cache (~/.cache/huggingface)?"; then
      envs/$env/bin/python -c "import sys; from huggingface_hub import snapshot_download as d; print(d(sys.argv[1]))" "$repo" \
        || warn "download of $repo failed — if the model is gated: accept its license on huggingface.co and run 'envs/$env/bin/hf auth login'"
    fi
  done
  if command -v lms >/dev/null; then
    VLM=$(.venv/bin/python -c "from sceneforge.core.config import load_config as l; print(l()['segment']['vlm_model'])")
    if confirm "Download VLM $VLM with LM Studio (lms get, ~6 GB for 8B 4-bit)?"; then
      lms get "$VLM" || warn "lms get failed; download $VLM from the LM Studio UI"
    fi
  else
    warn "LM Studio CLI (lms) not found; stage 2 will use the default class list without a VLM"
  fi
fi

say "done. Next:"
print "  source .venv/bin/activate"
print "  lms server start && lms load <vlm>      # for stage 2 inventory"
print "  sceneforge run '/path/clip1.mkv' '/path/clip2.mkv' --scene rumi_room"
