#!/usr/bin/env zsh
# SceneForge preflight: read-only inspection of the Mac and input clips.
# Installs nothing, modifies nothing. Writes a report to ./work/_doctor/report.txt
# Usage: scripts/doctor.sh [clip1.mkv clip2.mkv ...]

set -u
OUT_DIR="work/_doctor"
mkdir -p "$OUT_DIR"
REPORT="$OUT_DIR/report.txt"
: > "$REPORT"

log() { print -r -- "$*" | tee -a "$REPORT"; }
section() { log ""; log "=== $* ==="; }
have() { command -v "$1" >/dev/null 2>&1; }
ver() { if have "$1"; then log "$1: $("$@" 2>&1 | head -n 1)"; else log "$1: MISSING"; fi; }

section "System"
log "macOS: $(sw_vers -productVersion 2>/dev/null) ($(sw_vers -buildVersion 2>/dev/null))"
log "arch: $(uname -m)"
log "chip: $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
log "cores: perf=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null) eff=$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null)"
MEM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
log "unified memory: $((MEM_BYTES / 1024 / 1024 / 1024)) GB"
log "GPU: $(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Chipset Model|Total Number of Cores|Metal Support/ {printf "%s=%s; ", $1, $2}' | tr -s ' ')"
log "free disk (cwd): $(df -h . | awk 'NR==2 {print $4 " of " $2}')"
if [[ "$(uname -m)" != "arm64" ]]; then
  log "WARNING: shell is not arm64 (Rosetta?). Run Terminal natively."
fi

section "Toolchain"
ver brew --version
ver git --version
ver ffmpeg -version
ver ffprobe -version
ver cmake --version
ver uv --version
ver python3 --version
for p in python3.11 python3.12 python3.13; do have $p && log "$p: $($p -c 'import sys,platform;print(sys.version.split()[0], platform.machine())')"; done
ver colmap -h
ver glomap -h

section "Blender"
BLENDER=""
for c in /Applications/Blender.app/Contents/MacOS/Blender "$(command -v blender 2>/dev/null)"; do
  [[ -n "$c" && -x "$c" ]] && BLENDER="$c" && break
done
if [[ -n "$BLENDER" ]]; then
  log "blender: $BLENDER"
  log "$("$BLENDER" --version 2>/dev/null | head -n 1)"
else
  log "blender: NOT FOUND in /Applications or PATH"
fi

section "ComfyUI / LM Studio (optional backends)"
if curl -s -m 2 http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
  log "ComfyUI: running on :8188"; curl -s -m 2 http://127.0.0.1:8188/system_stats | head -c 600 | tee -a "$REPORT"; log ""
else
  log "ComfyUI: not running on :8188"
fi
if curl -s -m 2 http://127.0.0.1:1234/v1/models >/dev/null 2>&1; then
  log "LM Studio: running on :1234; models:"
  curl -s -m 2 http://127.0.0.1:1234/v1/models | tr ',' '\n' | grep '"id"' | tee -a "$REPORT"
else
  log "LM Studio: server not running on :1234"
fi
have lms && log "lms CLI: present"

section "PyTorch MPS (system python3, if torch installed)"
python3 - <<'PY' 2>&1 | tee -a "$REPORT"
try:
    import torch
    print("torch", torch.__version__, "mps available:", torch.backends.mps.is_available(), "built:", torch.backends.mps.is_built())
except Exception as e:
    print("torch not importable in system python3:", type(e).__name__)
PY

section "Input clips"
if (( $# == 0 )); then
  log "no clips passed"
fi
for clip in "$@"; do
  log "--- $clip"
  if [[ ! -f "$clip" ]]; then log "NOT FOUND"; continue; fi
  log "size: $(du -h "$clip" | cut -f1)"
  if have ffprobe; then
    ffprobe -v error -show_entries format=duration,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,pix_fmt,color_space,color_transfer,nb_frames \
      -of default=noprint_wrappers=1 "$clip" 2>&1 | tee -a "$REPORT"
    # Rough shot-cut count with ffmpeg's scene score (threshold 0.3), read-only.
    CUTS=$(ffmpeg -hide_banner -nostats -i "$clip" -an -vf "select='gt(scene,0.3)',showinfo" -f null - 2>&1 | grep -c 'pts_time')
    log "approx. shot cuts (scene>0.3): $CUTS"
  fi
done

log ""
log "Report saved to $REPORT"
