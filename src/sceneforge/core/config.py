"""Configuration loading: default TOML + optional user override, deep-merged."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(user_config: Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    with open(DEFAULT_CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    if user_config is not None:
        with open(user_config, "rb") as f:
            cfg = deep_merge(cfg, tomllib.load(f))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    if cfg.get("license_profile") not in ("noncommercial", "commercial"):
        raise ValueError(f"license_profile must be 'noncommercial' or 'commercial', got {cfg.get('license_profile')!r}")
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Resolve config paths relative to the repository root."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)
