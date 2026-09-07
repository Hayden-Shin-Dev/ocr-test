from __future__ import annotations

import json
import time
from threading import Lock
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from .config import settings
from .models import OCRDocument, OCRLine


def _to_python(value: Any) -> Any:
    """Convert Paddle/Numpy values into regular Python values."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if hasattr(value, "tolist"):
        return _to_python(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _to_python(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_to_python(item) for item in value]
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
    return [
        [float(point[0]), float(point[1])]
        for point in points
        if isinstance(point, Sequence) and len(point) >= 2
    ]


def parse_ocr_result(results: Any) -> list[OCRLine]:
    """Normalize PaddleOCR output into text, bbox and confidence lines only."""
    items = results if isinstance(results, Sequence) and not isinstance(results, (str, bytes, Mapping)) else [results]
    lines: list[OCRLine] = []
    for item in items:
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
    lines.sort(key=lambda line: (line.bbox["y0"], line.bbox["x0"]))
    return lines


class PaddleOCREngine:
    """Production OCR only: image -> OCR lines."""

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
        return OCRDocument(
            source_name=image_path.name,
            width=width,
            height=height,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            lines=lines,
        )


engine = PaddleOCREngine()
