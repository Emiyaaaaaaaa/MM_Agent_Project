"""Helpers for passing task images to multimodal LLM calls.

The UI-facing ContentBlock protocol is not the same as the model input
protocol. This module converts known task artifacts into LangChain-compatible
``image_url`` parts for nodes that explicitly need visual understanding.
"""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from backend.services.artifacts import project_root, resolve_artifact_path

ImageParts = List[Dict[str, Any]]
ImageMeta = List[Dict[str, Any]]


def _resolve_image_path(value: Any) -> Optional[Path]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.startswith("/plots/") or text.startswith("/exports/"):
        return resolve_artifact_path(text)

    path = Path(text)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(project_root() / path)
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _chart_id_from_item(item: Dict[str, Any], path: Path) -> str:
    explicit = str(item.get("chart_id") or item.get("id") or "").strip().lower()
    if explicit:
        return explicit
    filename = str(item.get("filename") or path.name).strip().lower()
    return filename.rsplit(".", 1)[0] if filename else ""


def _image_part_from_path(
    path: Path,
    *,
    label: str,
    source: str,
    chart_id: str = "",
    max_bytes: int = 8 * 1024 * 1024,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size <= 0 or size > max_bytes:
        return None

    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    if not mime_type.startswith("image/"):
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data:
        return None

    encoded = base64.b64encode(data).decode("utf-8")
    part = {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }
    meta = {
        "label": label,
        "source": source,
        "chart_id": chart_id,
        "path": str(path),
        "filename": path.name,
        "mime_type": mime_type,
        "size": size,
    }
    return part, meta


def _iter_artifact_items(shared_mem: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    manifest = shared_mem.get("artifacts_manifest")
    if isinstance(manifest, list):
        for item in manifest:
            if isinstance(item, dict):
                yield item


def _iter_rag_image_items(shared_mem: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    meta = shared_mem.get("fusion_rag_image_meta")
    if isinstance(meta, list):
        for item in meta:
            if isinstance(item, dict):
                yield item


def _iter_content_block_images(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, dict):
        if str(value.get("type", "")).lower() == "image":
            yield value
            return
        for child in value.values():
            yield from _iter_content_block_images(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_content_block_images(child)


def _iter_shared_content_block_items(shared_mem: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    keys = (
        "raw_document_content",
        "analysis_report",
        "mathematical_model",
        "paper_draft",
        "draft",
    )
    for key in keys:
        yield from _iter_content_block_images(shared_mem.get(key))


def collect_multimodal_images(
    shared_mem: Dict[str, Any],
    *,
    sources: Sequence[str] = ("artifacts",),
    chart_ids: Optional[Sequence[str]] = None,
    max_images: int = 6,
    max_bytes: int = 8 * 1024 * 1024,
) -> Tuple[ImageParts, ImageMeta]:
    """Collect image parts from shared memory.

    ``sources`` may include ``artifacts``, ``rag``, and/or ``blocks``.
    ``chart_ids`` filters task artifacts and image blocks by planned chart ids;
    RAG references are not chart-specific.
    """
    allowed_chart_ids = {
        str(item).strip().lower()
        for item in (chart_ids or [])
        if str(item).strip()
    }
    parts: ImageParts = []
    metas: ImageMeta = []
    seen_paths: set[str] = set()

    def append_image(raw_path: Any, source: str, label: str, chart_id: str = "") -> None:
        if len(parts) >= max_images:
            return
        resolved = _resolve_image_path(raw_path)
        if not resolved:
            return
        try:
            key = str(resolved.resolve())
        except OSError:
            key = str(resolved)
        if key in seen_paths:
            return
        seen_paths.add(key)
        built = _image_part_from_path(
            resolved,
            label=label,
            source=source,
            chart_id=chart_id,
            max_bytes=max_bytes,
        )
        if not built:
            return
        part, meta = built
        parts.append(part)
        metas.append(meta)

    if "rag" in sources:
        for idx, item in enumerate(_iter_rag_image_items(shared_mem), start=1):
            raw_path = item.get("path") or item.get("url")
            label = str(item.get("label") or f"rag_figure_{idx}")
            append_image(raw_path, "rag", label)
            if len(parts) >= max_images:
                break

    if "blocks" in sources and len(parts) < max_images:
        for idx, item in enumerate(_iter_shared_content_block_items(shared_mem), start=1):
            raw_path = item.get("path") or item.get("url")
            chart_id = str(item.get("chart_id") or item.get("id") or "").strip().lower()
            if allowed_chart_ids and chart_id and chart_id not in allowed_chart_ids:
                continue
            label = str(item.get("alt") or item.get("label") or chart_id or f"content_image_{idx}")
            append_image(raw_path, "content_block", label, chart_id=chart_id)
            if len(parts) >= max_images:
                break

    if "artifacts" in sources and len(parts) < max_images:
        for idx, item in enumerate(_iter_artifact_items(shared_mem), start=1):
            if str(item.get("kind", "")).lower() != "image":
                continue
            raw_path = item.get("path") or item.get("url")
            resolved = _resolve_image_path(raw_path)
            if not resolved:
                continue
            chart_id = _chart_id_from_item(item, resolved)
            if allowed_chart_ids and chart_id not in allowed_chart_ids:
                continue
            label = str(item.get("label") or chart_id or f"artifact_figure_{idx}")
            append_image(raw_path, "artifact", label, chart_id=chart_id)
            if len(parts) >= max_images:
                break

    return parts, metas


def image_meta_summary(meta: Sequence[Dict[str, Any]]) -> str:
    if not meta:
        return "No images attached."
    lines = []
    for item in meta:
        label = item.get("label") or item.get("filename") or "image"
        chart_id = item.get("chart_id") or "n/a"
        source = item.get("source") or "unknown"
        filename = item.get("filename") or ""
        lines.append(f"- {label}: source={source}, chart_id={chart_id}, filename={filename}")
    return "\n".join(lines)
