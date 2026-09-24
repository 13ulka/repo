"""Minimal HTML reports and contact sheets for visual checks."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


def contact_sheet(images: Sequence[np.ndarray], labels: Sequence[str], out: Path, cols: int = 6, thumb_w: int = 320) -> None:
    """Grid of BGR images with labels, written as JPEG."""
    if not images:
        return
    thumbs = []
    for img, label in zip(images, labels):
        h, w = img.shape[:2]
        th = int(round(h * thumb_w / w))
        t = cv2.resize(img, (thumb_w, th), interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (thumb_w, 18), (0, 0, 0), -1)
        cv2.putText(t, label[:48], (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(t)
    th = max(t.shape[0] for t in thumbs)
    thumbs = [np.pad(t, ((0, th - t.shape[0]), (0, 0), (0, 0))) for t in thumbs]
    rows = []
    for i in range(0, len(thumbs), cols):
        row = thumbs[i : i + cols]
        row += [np.zeros_like(thumbs[0])] * (cols - len(row))
        rows.append(np.concatenate(row, axis=1))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 88])


def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha: float = 0.5) -> np.ndarray:
    out = img_bgr.copy()
    m = mask.astype(bool)
    out[m] = (out[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def write_html(path: Path, title: str, sections: Sequence[tuple[str, str]]) -> None:
    """sections: (heading, raw html body)."""
    body = "".join(f"<h2>{html.escape(h)}</h2>{b}" for h, b in sections)
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font:14px -apple-system,system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}}
table{{border-collapse:collapse;margin:8px 0}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}
th{{background:#eee}}img{{max-width:100%;border:1px solid #ccc}}.warn{{color:#a40}}
@media (prefers-color-scheme: dark){{body{{background:#1b1b1b;color:#ddd}}th{{background:#333}}td,th{{border-color:#555}}}}
</style></head><body><h1>{html.escape(title)}</h1>{body}</body></html>"""
    path.write_text(doc)


def img_tag(rel: str) -> str:
    return f'<p><img src="{html.escape(rel)}"></p>'


def warnings_html(warnings: Sequence[str]) -> str:
    if not warnings:
        return "<p>none</p>"
    return "<ul>" + "".join(f'<li class="warn">{html.escape(w)}</li>' for w in warnings) + "</ul>"
