from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

from PIL import Image

from .config import settings


@dataclass(frozen=True)
class StructureBlock:
    label: str
    score: float
    bbox: dict[str, float]
    content: str = ""
    page_no: int = 0


@dataclass(frozen=True)
class StructureCell:
    bbox: dict[str, float]
    row_index: int
    column_index: int
    text: str = ""
    page_no: int = 0


@dataclass(frozen=True)
class StructureTable:
    bbox: dict[str, float]
    cells: list[StructureCell]
    html: str = ""
    score: float = 0.0
    page_no: int = 0


@dataclass(frozen=True)
class StructureDocument:
    width: int
    height: int
    blocks: list[StructureBlock]
    tables: list[StructureTable]
    reading_order: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _python(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if hasattr(value, "tolist"):
        return _python(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _python(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_python(item) for item in value]
    return value


def _result_payload(result: Any) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        payload: Any = result
    else:
        payload = getattr(result, "json", None)
        if callable(payload):
            payload = payload()
        if payload is None:
            payload = getattr(result, "to_dict", lambda: {})()
        payload = _python(payload)
    payload = _python(payload)
    if not isinstance(payload, Mapping):
        return {}
    nested = payload.get("res")
    return nested if isinstance(nested, Mapping) else payload


def _bbox(value: Any) -> dict[str, float] | None:
    points = _python(value)
    if isinstance(points, Mapping):
        if all(key in points for key in ("x0", "y0", "x1", "y1")):
            return {key: float(points[key]) for key in ("x0", "y0", "x1", "y1")}
        points = points.get("coordinate") or points.get("bbox")
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        return None
    if len(points) == 4 and all(isinstance(item, (int, float)) for item in points):
        return {"x0": float(points[0]), "y0": float(points[1]), "x1": float(points[2]), "y1": float(points[3])}
    flattened = [point for point in points if isinstance(point, Sequence) and len(point) >= 2]
    if len(flattened) < 2:
        return None
    return {
        "x0": min(float(point[0]) for point in flattened),
        "y0": min(float(point[1]) for point in flattened),
        "x1": max(float(point[0]) for point in flattened),
        "y1": max(float(point[1]) for point in flattened),
    }


def _union(boxes: list[dict[str, float]]) -> dict[str, float]:
    return {
        "x0": min(box["x0"] for box in boxes),
        "y0": min(box["y0"] for box in boxes),
        "x1": max(box["x1"] for box in boxes),
        "y1": max(box["y1"] for box in boxes),
    }


def _intersection_ratio(first: dict[str, float], second: dict[str, float]) -> float:
    x0 = max(first["x0"], second["x0"])
    y0 = max(first["y0"], second["y0"])
    x1 = min(first["x1"], second["x1"])
    y1 = min(first["y1"], second["y1"])
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    denominator = max(1.0, min(
        (first["x1"] - first["x0"]) * (first["y1"] - first["y0"]),
        (second["x1"] - second["x0"]) * (second["y1"] - second["y0"]),
    ))
    return area / denominator


def _axis_clusters(values: list[float], tolerance: float) -> list[float]:
    centers: list[float] = []
    for value in sorted(values):
        if not centers or abs(value - centers[-1]) > tolerance:
            centers.append(value)
        else:
            centers[-1] = (centers[-1] + value) / 2
    return centers


def _table_from_payload(payload: Mapping[str, Any], page_no: int, layout_blocks: list[StructureBlock]) -> list[StructureTable]:
    raw_tables = _python(payload.get("table_res_list", []))
    if not isinstance(raw_tables, Sequence):
        return []
    tables: list[StructureTable] = []
    for raw in raw_tables:
        if not isinstance(raw, Mapping):
            continue
        raw_boxes = _python(raw.get("cell_box_list", []))
        boxes = [_bbox(item) for item in raw_boxes] if isinstance(raw_boxes, Sequence) else []
        boxes = [box for box in boxes if box]
        if not boxes:
            continue
        tolerance = statistics.median(max(1.0, box["y1"] - box["y0"]) for box in boxes) * 0.55
        row_centers = _axis_clusters([(box["y0"] + box["y1"]) / 2 for box in boxes], tolerance)
        col_centers = _axis_clusters([box["x0"] for box in boxes], tolerance)
        cells: list[StructureCell] = []
        for box in boxes:
            center_y = (box["y0"] + box["y1"]) / 2
            row_index = min(range(len(row_centers)), key=lambda index: abs(row_centers[index] - center_y))
            column_index = min(range(len(col_centers)), key=lambda index: abs(col_centers[index] - box["x0"]))
            cells.append(StructureCell(box, row_index, column_index, page_no=page_no))
        score = max((block.score for block in layout_blocks if block.label.casefold() == "table" and _intersection_ratio(block.bbox, _union(boxes)) >= 0.1), default=0.0)
        tables.append(StructureTable(_union(boxes), cells, str(raw.get("pred_html", "")), score, page_no))
    return tables


def parse_structure_result(results: Any, width: int, height: int) -> StructureDocument:
    items = results if isinstance(results, Sequence) and not isinstance(results, (str, bytes, Mapping)) else [results]
    blocks: list[StructureBlock] = []
    tables: list[StructureTable] = []
    for page_no, item in enumerate(items):
        payload = _result_payload(item)
        layout = payload.get("layout_det_res", {})
        raw_boxes = layout.get("boxes", []) if isinstance(layout, Mapping) else []
        page_blocks: list[StructureBlock] = []
        for raw in _python(raw_boxes):
            if not isinstance(raw, Mapping):
                continue
            box = _bbox(raw.get("coordinate"))
            if box:
                page_blocks.append(StructureBlock(str(raw.get("label", "unknown")), float(raw.get("score", 0.0)), box, page_no=page_no))
        blocks.extend(page_blocks)
        tables.extend(_table_from_payload(payload, page_no, page_blocks))
    reading_order = sorted(
        range(len(blocks)),
        key=lambda index: (blocks[index].page_no, blocks[index].bbox["y0"], blocks[index].bbox["x0"]),
    )
    return StructureDocument(width, height, blocks, tables, reading_order)


class PPStructureV3Engine:
    """PP-StructureV3 only: image -> block/table/cell layout."""

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._lock = Lock()

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            from paddleocr import PPStructureV3

            self._pipeline = PPStructureV3(
                lang=settings.paddle_lang,
                device=settings.paddle_device,
                use_doc_orientation_classify=True,
                use_doc_unwarping=True,
                use_textline_orientation=True,
                use_table_recognition=True,
                use_formula_recognition=False,
            )
        return self._pipeline

    def predict(self, image_path: Path) -> StructureDocument:
        with Image.open(image_path) as image:
            width, height = image.size
        with self._lock:
            results = list(self._get_pipeline().predict(str(image_path)))
        return parse_structure_result(results, width, height)


structure_engine = PPStructureV3Engine()
