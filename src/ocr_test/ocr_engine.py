from __future__ import annotations

import json
import re
from threading import Lock
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from .config import settings


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float
    polygon: list[list[float]]


@dataclass(frozen=True)
class OCRField:
    label: str
    value: str
    confidence: float
    line_numbers: list[int]
    bbox: list[float]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OCRDocument:
    source_name: str
    width: int
    height: int
    elapsed_ms: int
    lines: list[OCRLine]
    fields: list[OCRField]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def structured_text(self) -> str:
        return "\n".join(
            f"{field.label}: {field.value}" if field.value else field.label
            for field in self.fields
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["text"] = self.text
        result["structured_text"] = self.structured_text
        return result


class PaddleOCREngine:
    """Lazy, reusable PaddleOCR pipeline for general document images."""

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._lock = Lock()

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            from paddleocr import PaddleOCR

            self._pipeline = PaddleOCR(
                lang=settings.paddle_lang,
                device=settings.paddle_device,
                use_doc_orientation_classify=True,
                use_doc_unwarping=True,
                use_textline_orientation=True,
            )
        return self._pipeline

    def predict(self, image_path: Path) -> OCRDocument:
        started = time.perf_counter()
        with Image.open(image_path) as image:
            width, height = image.size

        with self._lock:
            raw_results = self._get_pipeline().predict(str(image_path))
            lines = parse_ocr_result(raw_results)
        fields = build_field_mappings(lines)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        return OCRDocument(
            source_name=image_path.name,
            width=width,
            height=height,
            elapsed_ms=elapsed_ms,
            lines=lines,
            fields=fields,
        )


def _to_python(value: Any) -> Any:
    """Convert numpy-like values and JSON strings into regular Python values."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _payload_from_result(result: Any) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        payload: Any = result
    else:
        payload = getattr(result, "json", None)
        if callable(payload):
            payload = payload()
        if payload is None:
            payload = getattr(result, "to_dict", lambda: {})()
        payload = _to_python(payload)

    if not isinstance(payload, Mapping):
        return {}
    nested = payload.get("res")
    return nested if isinstance(nested, Mapping) else payload


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return _to_python(payload[key])
    return []


def _polygon(value: Any) -> list[list[float]]:
    points = _to_python(value)
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        return []
    normalized: list[list[float]] = []
    for point in points:
        if isinstance(point, Sequence) and len(point) >= 2:
            normalized.append([float(point[0]), float(point[1])])
    return normalized


def parse_ocr_result(results: Any) -> list[OCRLine]:
    """Normalize PaddleOCR 3.x output into sorted text lines.

    The parser deliberately relies only on generic OCR output fields and does
    not contain rules for invoices, packing lists, bills of lading, or samples.
    """
    result_items = results if isinstance(results, Sequence) and not isinstance(results, (str, bytes, Mapping)) else [results]
    lines: list[OCRLine] = []

    for item in result_items:
        payload = _payload_from_result(item)
        texts = _first_present(payload, "rec_texts", "rec_text")
        scores = _first_present(payload, "rec_scores", "rec_score")
        polygons = _first_present(payload, "dt_polys", "rec_polys", "polys")
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
            continue

        for index, raw_text in enumerate(texts):
            text = str(raw_text).strip()
            if not text:
                continue
            score = float(scores[index]) if isinstance(scores, Sequence) and index < len(scores) else 0.0
            polygon = _polygon(polygons[index]) if isinstance(polygons, Sequence) and index < len(polygons) else []
            lines.append(OCRLine(text=text, confidence=round(score, 4), polygon=polygon))

    lines.sort(key=lambda line: (
        min((point[1] for point in line.polygon), default=0),
        min((point[0] for point in line.polygon), default=0),
    ))
    return lines


def _line_bbox(line: OCRLine) -> tuple[float, float, float, float]:
    xs = [point[0] for point in line.polygon]
    ys = [point[1] for point in line.polygon]
    return (
        min(xs, default=0),
        min(ys, default=0),
        max(xs, default=0),
        max(ys, default=0),
    )


def _same_visual_column(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> bool:
    first_left, _, first_right, _ = first
    second_left, _, second_right, _ = second
    overlap = max(0.0, min(first_right, second_right) - max(first_left, second_left))
    shortest_width = max(1.0, min(first_right - first_left, second_right - second_left))
    left_distance = abs(first_left - second_left)
    return overlap / shortest_width >= 0.6 or left_distance <= max(18.0, shortest_width * 0.25)


def _connected_vertically(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> bool:
    _, first_top, _, first_bottom = first
    _, second_top, _, second_bottom = second
    if second_top < first_top:
        first, second = second, first
        _, first_top, _, first_bottom = first
        _, second_top, _, second_bottom = second
    first_height = max(1.0, first_bottom - first_top)
    second_height = max(1.0, second_bottom - second_top)
    first_center = (first_top + first_bottom) / 2
    second_center = (second_top + second_bottom) / 2
    if abs(first_center - second_center) <= min(first_height, second_height) * 0.55:
        return False
    gap = second_top - first_bottom
    return gap <= max(28.0, min(first_height, second_height) * 2.2)


def _looks_like_field_label(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith(":") or ("(" in stripped and ")" in stripped):
        return True
    words = re.findall(r"[A-Za-z]+", stripped)
    generic_markers = {
        "ADDRESS", "AMOUNT", "BILL", "BOOKING", "CARRIER", "CONSIGNEE", "DATE",
        "DESCRIPTION", "DESTINATION", "DELIVERY", "FORWARDING", "FREIGHT",
        "INSTRUCTIONS", "INVOICE", "LIABILITY", "LOADING", "MOVEMENT",
        "MEASUREMENT", "NAME", "NUMBER", "NO", "PACKAGES", "PORT", "PRICE",
        "PARTICULARS", "QUANTITY", "REFERENCE", "REFERENCES", "ROUTING", "SHIPMENT",
        "SHIPPER", "TOTAL", "UNIT", "VALUE", "WEIGHT", "COUNTRY", "CARRIAGE",
        "INSURANCE", "DECLARED", "CHARGES", "RATES",
    }
    return len(stripped) <= 90 and any(word.upper().rstrip(".") in generic_markers for word in words)


def _split_inline_label(text: str) -> tuple[str, str]:
    if ":" not in text:
        return text, ""
    label, value = text.split(":", 1)
    if label.strip() and value.strip():
        return label.strip(), value.strip()
    return text.rstrip(":"), ""


def build_field_mappings(lines: list[OCRLine]) -> list[OCRField]:
    """Group OCR lines by visual layout without document-specific templates."""
    if not lines:
        return []

    boxes = [_line_bbox(line) for line in lines]
    blocks: list[list[int]] = []
    for index, box in enumerate(boxes):
        candidates: list[tuple[float, int]] = []
        for block_index, block in enumerate(blocks):
            previous_index = block[-1]
            previous_box = boxes[previous_index]
            starts_new_field = _looks_like_field_label(lines[index].text)
            if not starts_new_field and _same_visual_column(previous_box, box) and _connected_vertically(previous_box, box):
                candidates.append((box[1] - previous_box[3], block_index))
        if candidates:
            _, chosen_block = min(candidates)
            blocks[chosen_block].append(index)
        else:
            blocks.append([index])

    fields: list[OCRField] = []
    for block in blocks:
        block_lines = [lines[index] for index in block]
        label, inline_value = _split_inline_label(block_lines[0].text)
        value_lines = block_lines[1:]
        value_parts = [inline_value] if inline_value else []
        value_parts.extend(line.text for line in value_lines)
        value = " ".join(value_parts)
        confidence = sum(line.confidence for line in block_lines) / len(block_lines)
        block_boxes = [boxes[index] for index in block]
        fields.append(OCRField(
            label=label,
            value=value,
            confidence=round(confidence, 4),
            line_numbers=[index + 1 for index in block],
            bbox=[
                min(box[0] for box in block_boxes),
                min(box[1] for box in block_boxes),
                max(box[2] for box in block_boxes),
                max(box[3] for box in block_boxes),
            ],
        ))
    return fields


engine = PaddleOCREngine()
