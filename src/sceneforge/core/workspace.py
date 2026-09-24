"""Scene workspace layout: work/<scene>/<stage>/ with atomic stage outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import REPO_ROOT

STAGES = ["s0_ingest", "s0b_masks", "s1_geometry", "s2_segment"]


def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return s or "scene"


@dataclass
class Workspace:
    root: Path  # work/<scene>

    @classmethod
    def for_scene(cls, scene: str, work_dir: Path | None = None) -> "Workspace":
        base = work_dir or (REPO_ROOT / "work")
        return cls(base / slugify(scene))

    def stage_dir(self, stage: str) -> Path:
        return self.root / stage

    def record_path(self, stage: str) -> Path:
        return self.stage_dir(stage) / "_stage.json"

    def load_record(self, stage: str) -> dict[str, Any] | None:
        p = self.record_path(stage)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    def require(self, stage: str) -> dict[str, Any]:
        rec = self.load_record(stage)
        if not rec or rec.get("status") != "done":
            raise RuntimeError(f"Stage '{stage}' has no completed output in {self.root}. Run it first.")
        return rec

    @property
    def scene_file(self) -> Path:
        return self.root / "scene.json"

    def load_scene(self) -> dict[str, Any]:
        if not self.scene_file.exists():
            raise RuntimeError(f"No scene at {self.root}. Start with `sceneforge ingest <clips...> --scene NAME`.")
        return json.loads(self.scene_file.read_text())

    def save_scene(self, data: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.scene_file, data)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=_json_default))
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")
