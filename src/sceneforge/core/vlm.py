"""OpenAI-compatible VLM client (LM Studio / mlx_vlm.server) with JSON-schema output."""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any

import cv2
import numpy as np
import requests

log = logging.getLogger("sceneforge")


class VLMUnavailable(RuntimeError):
    pass


def encode_image(img_bgr: np.ndarray, max_side: int = 768) -> str:
    h, w = img_bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _extract_json(text: str) -> Any:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fence:
        text = fence.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError(f"no JSON in VLM response: {text[:200]!r}")
    return json.loads(text[start : max(text.rfind("}"), text.rfind("]")) + 1])


class VLMClient:
    def __init__(self, base_url: str, model: str, timeout_s: float = 600):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def check(self) -> list[str]:
        try:
            r = requests.get(f"{self.base_url}/models", timeout=5)
            r.raise_for_status()
        except requests.RequestException as e:
            raise VLMUnavailable(
                f"VLM server not reachable at {self.base_url} ({e}). Start it with:\n"
                f"  lms server start && lms load {self.model}"
            ) from e
        ids = [m.get("id") for m in r.json().get("data", [])]
        if self.model not in ids:
            log.warning("VLM model %s not listed by server (available: %s); LM Studio may JIT-load it", self.model, ids)
        return ids

    def ask_json(self, prompt: str, images_bgr: list[np.ndarray], schema: dict[str, Any], max_tokens: int = 2048) -> Any:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content += [{"type": "image_url", "image_url": {"url": encode_image(im)}} for im in images_bgr]
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}},
        }
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout_s)
        if r.status_code >= 400:
            # some servers reject response_format; retry with instructions only
            log.warning("VLM rejected json_schema (%s); retrying without it", r.status_code)
            body.pop("response_format")
            r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout_s)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        return _extract_json(text)
