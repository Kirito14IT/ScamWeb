"""Lightweight visual metadata analysis.

This demo intentionally avoids loading a large VLM on the local 8GB GPU.
It only records image availability and dimensions. Future VLM backends can use
the same sample schema and replace this module with richer screenshot analysis.
"""

from __future__ import annotations

from pathlib import Path

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None


def _read_png_size(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        return width, height
    return None, None


def get_image_size(path: str | None) -> tuple[int | None, int | None]:
    if not path:
        return None, None
    image_path = Path(path)
    if not image_path.exists():
        return None, None
    if Image is not None:
        try:
            with Image.open(image_path) as image:
                return image.size
        except Exception:
            # Some crawled screenshots are malformed or extremely large.
            # Keep the sample runnable; official filtering can handle these later.
            pass
    if image_path.suffix.lower() == ".png":
        return _read_png_size(image_path)
    return None, None


def analyze_visual_layout(image_path: str | None) -> dict:
    width, height = get_image_size(image_path)
    signals: list[str] = []
    if width and height and height > width * 1.8:
        signals.append("long_page_capture")
    return {
        "image_path": image_path,
        "image_exists": bool(image_path and Path(image_path).exists()),
        "width": width,
        "height": height,
        "risk_score": 0,
        "signals": signals,
    }
