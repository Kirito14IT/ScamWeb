"""Deterministic full-page overview/tiling for Qwen3-VL inputs."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


# The frozen benchmark contains trusted full-page captures up to roughly 442 MP.
# Keep source decoding possible while the submitted assets remain capped at 8 MP.
Image.MAX_IMAGE_PIXELS = 500_000_000
IMAGE_PREPROCESSING_VERSION = "contact-sheet-v3"


@dataclass(frozen=True)
class ImagePolicy:
    overview_pixels: int = 1_048_576
    total_pixels: int = 8_388_608
    max_long_side: int = 7680
    max_tiles: int = 7
    overlap_ratio: float = 0.10
    max_base64_bytes: int = 10_000_000
    contact_sheet_pixels: int = 8_388_608
    contact_sheet_base64_bytes: int = 2_500_000


@dataclass
class ImageAsset:
    index: int
    kind: str
    path: str
    width: int
    height: int
    crop_xyxy: list[float]
    source_size: list[int]
    mime_type: str
    data_url_size: int
    logical_indexes: list[int] = field(default_factory=list)

    def data_url(self) -> str:
        encoded = base64.b64encode(Path(self.path).read_bytes()).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"


@dataclass
class ImageBundle:
    source_path: str
    source_size: list[int]
    assets: list[ImageAsset]
    policy: dict[str, Any]
    contact_sheet: ImageAsset | None = None
    contact_sheet_panels: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_size": self.source_size,
            "assets": [asdict(asset) for asset in self.assets],
            "policy": self.policy,
            "contact_sheet": asdict(self.contact_sheet) if self.contact_sheet else None,
            "contact_sheet_panels": self.contact_sheet_panels,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ImageBundle":
        return cls(
            source_path=str(value["source_path"]),
            source_size=list(value["source_size"]),
            assets=[ImageAsset(**asset) for asset in value["assets"]],
            policy=dict(value["policy"]),
            contact_sheet=(
                ImageAsset(**value["contact_sheet"])
                if isinstance(value.get("contact_sheet"), dict)
                else None
            ),
            contact_sheet_panels=list(value.get("contact_sheet_panels") or []),
        )


def _target_size(width: int, height: int, pixel_budget: int, max_long_side: int) -> tuple[int, int]:
    scale = min(1.0, math.sqrt(pixel_budget / max(1, width * height)), max_long_side / max(width, height))
    target_width = max(1, int(math.floor(width * scale)))
    target_height = max(1, int(math.floor(height * scale)))
    while target_width * target_height > pixel_budget:
        if target_width >= target_height:
            target_width -= 1
        else:
            target_height -= 1
    return target_width, target_height


def _encode_image(image: Image.Image, path_stem: Path, max_base64_bytes: int) -> tuple[Path, str, int]:
    png_path = path_stem.with_suffix(".png")
    image.save(png_path, format="PNG", optimize=True)
    encoded_size = len(base64.b64encode(png_path.read_bytes()))
    if encoded_size <= max_base64_bytes:
        return png_path, "image/png", encoded_size

    working = image.convert("RGB")
    quality = 92
    while quality >= 50:
        jpg_path = path_stem.with_suffix(".jpg")
        working.save(jpg_path, format="JPEG", quality=quality, optimize=True)
        encoded_size = len(base64.b64encode(jpg_path.read_bytes()))
        if encoded_size <= max_base64_bytes:
            png_path.unlink(missing_ok=True)
            return jpg_path, "image/jpeg", encoded_size
        quality -= 7
    raise ValueError(f"Prepared image cannot satisfy Base64 limit: {path_stem}")


def _prepare_asset(
    source: Image.Image,
    crop: tuple[int, int, int, int],
    pixel_budget: int,
    max_long_side: int,
    cache_dir: Path,
    name: str,
    index: int,
    kind: str,
    max_base64_bytes: int,
) -> ImageAsset:
    cropped = source.crop(crop)
    target_size = _target_size(cropped.width, cropped.height, pixel_budget, max_long_side)
    if cropped.size != target_size:
        cropped = cropped.resize(target_size, Image.Resampling.LANCZOS)
    path, mime_type, data_url_size = _encode_image(cropped, cache_dir / name, max_base64_bytes)
    return ImageAsset(
        index=index,
        kind=kind,
        path=str(path),
        width=cropped.width,
        height=cropped.height,
        crop_xyxy=[float(value) for value in crop],
        source_size=[source.width, source.height],
        mime_type=mime_type,
        data_url_size=data_url_size,
    )


def _prepare_contact_sheet(
    assets: list[ImageAsset],
    cache_dir: Path,
    policy: ImagePolicy,
) -> tuple[ImageAsset, list[dict[str, Any]]]:
    """Pack logical overview/tile images into one provider-compatible atlas."""

    columns = min(4, len(assets))
    rows = math.ceil(len(assets) / columns)
    content_width = 480
    content_height = 1920
    header_height = 44
    padding = 10
    cell_width = content_width + padding * 2
    cell_height = content_height + header_height + padding * 2
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(sheet)
    except Exception:  # pragma: no cover - Pillow always supplies ImageDraw.
        draw = None
    panels: list[dict[str, Any]] = []
    for position, asset in enumerate(assets):
        row, column = divmod(position, columns)
        cell_x = column * cell_width
        cell_y = row * cell_height
        if draw is not None:
            draw.rectangle(
                [cell_x, cell_y, cell_x + cell_width - 1, cell_y + header_height - 1],
                fill=(22, 31, 43),
            )
            draw.text(
                (cell_x + padding, cell_y + 12),
                f"image_index {asset.index} | {asset.kind}",
                fill="white",
            )
        with Image.open(asset.path) as opened:
            panel_image = opened.convert("RGB")
        scale = min(
            1.0,
            content_width / max(1, panel_image.width),
            content_height / max(1, panel_image.height),
        )
        target = (
            max(1, int(math.floor(panel_image.width * scale))),
            max(1, int(math.floor(panel_image.height * scale))),
        )
        if panel_image.size != target:
            panel_image = panel_image.resize(target, Image.Resampling.LANCZOS)
        content_x = cell_x + padding + (content_width - panel_image.width) // 2
        content_y = cell_y + header_height + padding + (content_height - panel_image.height) // 2
        sheet.paste(panel_image, (content_x, content_y))
        if draw is not None:
            draw.rectangle(
                [content_x, content_y, content_x + panel_image.width - 1, content_y + panel_image.height - 1],
                outline=(220, 38, 38),
                width=3,
            )
        panels.append(
            {
                "image_index": asset.index,
                "kind": asset.kind,
                "content_xyxy": [
                    float(content_x),
                    float(content_y),
                    float(content_x + panel_image.width),
                    float(content_y + panel_image.height),
                ],
            }
        )

    target_size = _target_size(
        sheet.width,
        sheet.height,
        policy.contact_sheet_pixels,
        policy.max_long_side,
    )
    if sheet.size != target_size:
        scale_x = target_size[0] / sheet.width
        scale_y = target_size[1] / sheet.height
        sheet = sheet.resize(target_size, Image.Resampling.LANCZOS)
        for panel in panels:
            x1, y1, x2, y2 = panel["content_xyxy"]
            panel["content_xyxy"] = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
    path, mime_type, data_url_size = _encode_image(
        sheet,
        cache_dir / "contact_sheet",
        policy.contact_sheet_base64_bytes,
    )
    return (
        ImageAsset(
            index=0,
            kind="contact_sheet",
            path=str(path),
            width=sheet.width,
            height=sheet.height,
            crop_xyxy=[0.0, 0.0, float(sheet.width), float(sheet.height)],
            source_size=list(assets[0].source_size),
            mime_type=mime_type,
            data_url_size=data_url_size,
            logical_indexes=[asset.index for asset in assets],
        ),
        panels,
    )


def prepare_image_bundle(source_path: str | Path, cache_root: str | Path, policy: ImagePolicy | None = None) -> ImageBundle:
    policy = policy or ImagePolicy()
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    fingerprint = hashlib.sha256(
        f"{source_path.resolve()}:{source_path.stat().st_size}:{source_path.stat().st_mtime_ns}:"
        f"{IMAGE_PREPROCESSING_VERSION}:{json.dumps(asdict(policy), sort_keys=True)}".encode("utf-8")
    ).hexdigest()[:20]
    cache_dir = Path(cache_root) / fingerprint
    manifest_path = cache_dir / "bundle.json"
    if manifest_path.exists():
        bundle = ImageBundle.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
        if all(Path(asset.path).exists() for asset in bundle.assets):
            return bundle
    cache_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as opened:
        source = opened.convert("RGB")
    width, height = source.size
    is_long = height > policy.max_long_side or width > policy.max_long_side or max(width, height) / max(1, min(width, height)) > 4
    assets: list[ImageAsset] = []
    if not is_long:
        assets.append(
            _prepare_asset(
                source,
                (0, 0, width, height),
                min(policy.overview_pixels, policy.total_pixels),
                policy.max_long_side,
                cache_dir,
                "overview",
                0,
                "overview",
                policy.max_base64_bytes,
            )
        )
    else:
        assets.append(
            _prepare_asset(
                source,
                (0, 0, width, height),
                min(policy.overview_pixels, policy.total_pixels),
                policy.max_long_side,
                cache_dir,
                "overview",
                0,
                "overview",
                policy.max_base64_bytes,
            )
        )
        ratio = height / max(1, width)
        tile_count = min(
            policy.max_tiles,
            max(2, math.ceil(ratio / 4), math.ceil(height / max(1, policy.max_long_side))),
        )
        remaining = max(1, policy.total_pixels - assets[0].width * assets[0].height)
        tile_budget = max(1, remaining // tile_count)
        segment = height / tile_count
        overlap = segment * policy.overlap_ratio
        for tile_index in range(tile_count):
            top = max(0, math.floor(tile_index * segment - (overlap if tile_index else 0)))
            bottom = min(height, math.ceil((tile_index + 1) * segment + (overlap if tile_index + 1 < tile_count else 0)))
            assets.append(
                _prepare_asset(
                    source,
                    (0, top, width, bottom),
                    tile_budget,
                    policy.max_long_side,
                    cache_dir,
                    f"tile_{tile_index + 1}",
                    tile_index + 1,
                    "tile",
                    policy.max_base64_bytes,
                )
            )
    contact_sheet = None
    contact_sheet_panels: list[dict[str, Any]] = []
    if len(assets) > 1:
        contact_sheet, contact_sheet_panels = _prepare_contact_sheet(assets, cache_dir, policy)
    bundle = ImageBundle(
        source_path=str(source_path),
        source_size=[width, height],
        assets=assets,
        policy={**asdict(policy), "preprocessing_version": IMAGE_PREPROCESSING_VERSION},
        contact_sheet=contact_sheet,
        contact_sheet_panels=contact_sheet_panels,
    )
    manifest_path.write_text(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return bundle


def map_bbox_1000(asset: ImageAsset, bbox_1000: list[float]) -> list[float]:
    if not isinstance(bbox_1000, (list, tuple)) or len(bbox_1000) != 4:
        raise ValueError("bbox_1000 must contain four coordinates")
    values = [max(0.0, min(1000.0, float(value))) for value in bbox_1000]
    left, top, right, bottom = asset.crop_xyxy
    crop_width = right - left
    crop_height = bottom - top
    mapped = [
        left + values[0] / 1000.0 * crop_width,
        top + values[1] / 1000.0 * crop_height,
        left + values[2] / 1000.0 * crop_width,
        top + values[3] / 1000.0 * crop_height,
    ]
    if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
        raise ValueError("bbox_1000 is degenerate")
    return mapped


def assets_for_stage(bundle: ImageBundle, stage: str) -> list[ImageAsset]:
    if stage in {"visual", "grounding"}:
        if bundle.contact_sheet is not None:
            return [bundle.contact_sheet]
        return list(bundle.assets)
    if stage in {"cross", "binary"}:
        return [bundle.assets[0]]
    if stage == "type":
        return []
    return []


def normalize_grounding_boxes(result: dict[str, Any], bundle: ImageBundle) -> dict[str, Any]:
    normalized = dict(result)
    boxes = []
    by_index = {asset.index: asset for asset in bundle.assets}
    for box in result.get("boxes") or []:
        if not isinstance(box, dict):
            boxes.append(box)
            continue
        converted = dict(box)
        if "bbox_1000" in converted:
            try:
                image_index = int(converted.get("image_index", 0))
                asset = by_index[image_index]
                converted["xyxy"] = map_bbox_1000(asset, converted["bbox_1000"])
            except (KeyError, TypeError, ValueError):
                converted["xyxy"] = None
        boxes.append(converted)
    normalized["boxes"] = boxes
    return normalized
