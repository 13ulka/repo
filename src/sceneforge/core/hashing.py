"""Stable hashing helpers for stage cache keys."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_json(obj: Any) -> str:
    return hash_bytes(json.dumps(obj, sort_keys=True, default=str).encode())


def hash_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def hash_files(paths: list[Path]) -> str:
    return hash_json([[str(p.name), hash_file(p)] for p in sorted(paths)])
