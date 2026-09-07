from __future__ import annotations

import json
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
class OCRDocument:
    source_name: str
    width: int
    height: int
    elapsed_ms: int
    lines: list[OCRLine]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["text"] = self.text
        return result


class PaddleOCREngine:
    """Lazy, reusable PaddleOCR pipeline for general document images."""

    def __init__(self) -> None:
        self._pipeline: Any | None = None

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

        raw_results = self._get_pipeline().predict(str(image_path))
        lines = parse_ocr_result(raw_results)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        return OCRDocument(
            source_name=image_path.name,
            width=width,
            height=height,
            elapsed_ms=elapsed_ms,
            lines=lines,
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


engine = PaddleOCREngine()
