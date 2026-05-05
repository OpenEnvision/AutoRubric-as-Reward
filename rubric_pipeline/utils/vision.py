# -*- coding: utf-8 -*-
"""Utilities for vision Auto-Rubric inputs.

The Auto-Rubric code accepts a few dataset shapes because the training and
benchmark scripts were written at different times.  This module keeps the
normalization rules in one place so rubric generation and final grading see the
same image order.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Iterable, Mapping


TASK_T2I = "t2i"
TASK_IMAGE_EDIT = "image_edit"


def normalize_task_type(task_type: str | None) -> str:
    """Normalize common task aliases to the two supported vision task types."""
    value = (task_type or TASK_T2I).strip().lower().replace("-", "_")
    if value in {"t2i", "text_to_image", "text2image", "image_generation", "gen"}:
        return TASK_T2I
    if value in {"edit", "image_edit", "image_editing", "i2i", "image_to_image"}:
        return TASK_IMAGE_EDIT
    raise ValueError(f"Unsupported vision Auto-Rubric task type: {task_type}")


def is_image_edit_task(task_type: str | None) -> bool:
    return normalize_task_type(task_type) == TASK_IMAGE_EDIT


def _as_path_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, Iterable):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _first_path_from_keys(data: Mapping[str, Any], keys: tuple[str, ...]) -> list[str]:
    for key in keys:
        paths = _as_path_list(data.get(key))
        if paths:
            return paths
    return []


def split_image_paths(data: Mapping[str, Any], task_type: str | None) -> tuple[str | None, list[str]]:
    """Split a sample into optional base image and candidate output images.

    T2I:
        returns (None, generated image paths).

    Image editing:
        returns (base image if present, edited output image paths).

    Supported edit datasets are not completely uniform.  Prefer explicit base
    and edited-image keys.  When only response/images are present, infer the old
    "response=[base, edit1, edit2...]" layout only when labels make that clear,
    or for unlabeled edit samples with at least three images.
    """
    task = normalize_task_type(task_type)

    if task == TASK_T2I:
        outputs = _first_path_from_keys(
            data,
            ("image_paths", "response", "responses", "images", "image", "generated_image", "generated_images"),
        )
        return None, outputs

    base_paths = _first_path_from_keys(
        data,
        ("source_image", "base_image", "original_image", "input_image", "reference_image", "ref_image"),
    )

    explicit_edited_paths = _first_path_from_keys(
        data,
        (
            "edited_image",
            "edited_images",
            "target_image",
            "target_images",
            "output_image",
            "output_images",
            "result_image",
            "result_images",
        ),
    )

    if base_paths:
        if explicit_edited_paths:
            return base_paths[0], explicit_edited_paths
        response_paths = _first_path_from_keys(data, ("response", "responses", "images", "image_paths", "image"))
        return base_paths[0], response_paths

    if explicit_edited_paths:
        return None, explicit_edited_paths

    response_paths = _first_path_from_keys(data, ("response", "responses", "images", "image_paths", "image"))
    if not response_paths:
        return None, []

    label_rank = data.get("label_rank")
    if isinstance(label_rank, Iterable) and not isinstance(label_rank, (str, bytes, Mapping)):
        expected_outputs = len(list(label_rank))
        if len(response_paths) == expected_outputs + 1:
            return response_paths[0], response_paths[1:]
        return None, response_paths

    if "label_score" in data and len(response_paths) == 2:
        return response_paths[0], response_paths[1:]

    if bool(data.get("response_includes_base")) and len(response_paths) >= 2:
        return response_paths[0], response_paths[1:]

    if len(response_paths) >= 3:
        return response_paths[0], response_paths[1:]

    return None, response_paths


def has_base_image(data: Mapping[str, Any], task_type: str | None) -> bool:
    base_path, _ = split_image_paths(data, task_type)
    return base_path is not None


def extract_image_paths(data: Mapping[str, Any], task_type: str | None) -> list[str]:
    """Extract image paths in prompt order: optional base image, then outputs."""
    base_path, output_paths = split_image_paths(data, task_type)
    if base_path:
        return [base_path, *output_paths]
    return output_paths


def count_outputs(
    image_paths: list[str] | Mapping[str, Any],
    task_type: str | None,
    has_base: bool | None = None,
) -> int:
    """Count candidate outputs, excluding the source image for editing tasks."""
    if isinstance(image_paths, Mapping):
        _, output_paths = split_image_paths(image_paths, task_type)
        return len(output_paths)
    if is_image_edit_task(task_type) and has_base:
        return max(0, len(image_paths) - 1)
    return len(image_paths)


def image_to_base64_url(path: str) -> str:
    """Convert a local image path to a data URL, preserving HTTP/data URLs."""
    if path.startswith(("http://", "https://", "data:")):
        return path

    image_path = Path(path).expanduser()
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "image/png"

    with image_path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def add_images_to_messages(
    messages: list[dict[str, Any]],
    image_paths: list[str],
    task_type: str | None,
    has_base: bool | None = None,
) -> list[dict[str, Any]]:
    """Attach images to the final user message using OpenAI vision content blocks."""
    if not image_paths:
        return messages

    task = normalize_task_type(task_type)
    content_text = messages[-1]["content"]
    image_urls = [image_to_base64_url(path) for path in image_paths]
    blocks: list[dict[str, Any]] = []

    if has_base is None:
        has_base = task == TASK_IMAGE_EDIT and len(image_urls) >= 3

    if task == TASK_IMAGE_EDIT and has_base and len(image_urls) > 1:
        blocks.extend(
            [
                {"type": "text", "text": "Image BASE (original image):"},
                {"type": "image_url", "image_url": {"url": image_urls[0]}},
            ],
        )
        edited_urls = image_urls[1:]
        if len(edited_urls) == 1:
            blocks.append({"type": "text", "text": "Edited Image:"})
            blocks.append({"type": "image_url", "image_url": {"url": edited_urls[0]}})
        else:
            for idx, url in enumerate(edited_urls, start=1):
                blocks.append({"type": "text", "text": f"Edited Image {idx}:"})
                blocks.append({"type": "image_url", "image_url": {"url": url}})
    elif task == TASK_IMAGE_EDIT:
        for idx, url in enumerate(image_urls, start=1):
            label = "Edited Image:" if len(image_urls) == 1 else f"Edited Image {idx}:"
            blocks.append({"type": "text", "text": label})
            blocks.append({"type": "image_url", "image_url": {"url": url}})
    else:
        for idx, url in enumerate(image_urls, start=1):
            label = "Generated Image:" if len(image_urls) == 1 else f"Image {idx}:"
            blocks.append({"type": "text", "text": label})
            blocks.append({"type": "image_url", "image_url": {"url": url}})

    blocks.append({"type": "text", "text": content_text})
    messages[-1]["content"] = blocks
    return messages
