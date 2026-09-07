from __future__ import annotations

import json
import statistics
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
    page_no: int = 0

    @property
    def bbox(self) -> dict[str, float]:
        x0, y0, x1, y1 = _line_bbox(self)
        return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "polygon": self.polygon,
            "page_no": self.page_no,
        }


@dataclass(frozen=True)
class OCRField:
    label: str
    value: str | list[str]
    confidence: float
    raw_lines: list[dict[str, Any]]
    bbox: dict[str, float]
    mapping_method: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OCRMapping:
    fields: list[OCRField]
    layout_mode: str
    low_confidence_count: int


@dataclass(frozen=True)
class OCRDocument:
    source_name: str
    width: int
    height: int
    elapsed_ms: int
    lines: list[OCRLine]
    fields: list[OCRField]
    layout_mode: str
    low_confidence_count: int

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def structured_text(self) -> str:
        def format_value(value: str | list[str]) -> str:
            if isinstance(value, list):
                return " | ".join(item for item in value if item)
            return value

        return "\n".join(
            f"{field.label}: {format_value(field.value)}" if field.value else field.label
            for field in self.fields
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["text"] = self.text
        result["structured_text"] = self.structured_text
        result["raw_text_lines"] = [line.as_dict() for line in self.lines]
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
        mapping = map_ocr_lines(lines, image_path=image_path)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        return OCRDocument(
            source_name=image_path.name,
            width=width,
            height=height,
            elapsed_ms=elapsed_ms,
            lines=lines,
            fields=mapping.fields,
            layout_mode=mapping.layout_mode,
            low_confidence_count=mapping.low_confidence_count,
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


GAP_X_MULTIPLIER = 2.5
GAP_Y_MULTIPLIER = 1.8
KNOWN_LABEL_TERMS = {
    "ADDRESS", "AMOUNT", "BILL", "BOOKING", "CARRIER", "CONSIGNEE", "DATE",
    "DESCRIPTION", "DESTINATION", "DELIVERY", "EXPORT", "FORWARDING", "FREIGHT",
    "INSTRUCTIONS", "INVOICE", "LIABILITY", "LOADING", "MEASUREMENT", "NAME",
    "NUMBER", "NO", "NOTIFY", "ORIGIN", "PACKAGES", "PARTICULARS", "PORT",
    "PRICE", "QUANTITY", "REFERENCE", "SHIPMENT", "SHIPPER", "TOTAL", "UNIT",
    "VALUE", "WEIGHT", "INSURANCE", "DECLARED", "CHARGES", "RATES", "ROUTING",
    "MARKS", "PAYABLE", "RECEIPT", "MOVEMENT", "COUNTRY",
}


def _median_or(values: list[float], fallback: float) -> float:
    return statistics.median(values) if values else fallback


def _vertical_overlap(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    overlap = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return overlap / max(1.0, min(first[3] - first[1], second[3] - second[1]))


def _horizontal_overlap(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    return overlap / max(1.0, min(first[2] - first[0], second[2] - second[0]))


def _rough_rows(lines: list[OCRLine], average_height: float) -> list[list[int]]:
    rows: list[list[int]] = []
    for index in sorted(range(len(lines)), key=lambda item: (_line_bbox(lines[item])[1], _line_bbox(lines[item])[0])):
        box = _line_bbox(lines[index])
        center_y = (box[1] + box[3]) / 2
        matching = []
        for row_index, row in enumerate(rows):
            row_boxes = [_line_bbox(lines[item]) for item in row]
            row_center = statistics.mean((item[1] + item[3]) / 2 for item in row_boxes)
            if abs(center_y - row_center) <= average_height * 0.65:
                matching.append(row_index)
        if matching:
            rows[matching[0]].append(index)
        else:
            rows.append([index])
    for row in rows:
        row.sort(key=lambda item: _line_bbox(lines[item])[0])
    return rows


def _local_gaps(lines: list[OCRLine]) -> tuple[float, float, float]:
    boxes = [_line_bbox(line) for line in lines]
    heights = [max(1.0, box[3] - box[1]) for box in boxes]
    average_height = statistics.median(heights) if heights else 1.0
    rows = _rough_rows(lines, average_height)
    horizontal_gaps: list[float] = []
    for row in rows:
        for first_index, second_index in zip(row, row[1:]):
            gap = boxes[second_index][0] - boxes[first_index][2]
            if gap > 0:
                horizontal_gaps.append(gap)

    vertical_gaps: list[float] = []
    ordered = sorted(range(len(lines)), key=lambda item: _line_bbox(lines[item])[1])
    for current_position, current_index in enumerate(ordered):
        current = boxes[current_index]
        prior_candidates = []
        for previous_index in ordered[:current_position]:
            previous = boxes[previous_index]
            if _horizontal_overlap(previous, current) >= 0.2:
                gap = current[1] - previous[3]
                if gap >= 0:
                    prior_candidates.append(gap)
        if prior_candidates:
            vertical_gaps.append(min(prior_candidates))

    vertical_baseline = _median_or(vertical_gaps, average_height)
    if 0 < len(vertical_gaps) <= 3:
        # With only a few vertical relationships, the median can be pulled
        # toward a section break. The closest local relationship is the safer
        # baseline for hierarchical grouping.
        vertical_baseline = min(vertical_gaps)
    return (_median_or(horizontal_gaps, average_height), vertical_baseline, average_height)


def _horizontal_cell_rows(
    lines: list[OCRLine],
    rows: list[list[int]],
    gap_x: float,
    average_height: float,
) -> list[list[int]]:
    boxes = [_line_bbox(line) for line in lines]
    cell_rows: list[list[list[int]]] = []
    # A page containing only one fragment per column has no small within-cell
    # gaps to establish a useful median. Keep the median rule, with a local
    # text-height cap for that sparse case; this remains resolution-independent.
    threshold = min(gap_x * GAP_X_MULTIPLIER, average_height * 2.0)
    for row in rows:
        cell = [row[0]]
        row_cells: list[list[int]] = []
        for previous_index, current_index in zip(row, row[1:]):
            gap = boxes[current_index][0] - boxes[previous_index][2]
            if gap >= threshold:
                row_cells.append(cell)
                cell = [current_index]
            else:
                cell.append(current_index)
        row_cells.append(cell)
        cell_rows.append(row_cells)
    return cell_rows


def _horizontal_cells(
    lines: list[OCRLine],
    rows: list[list[int]],
    gap_x: float,
    average_height: float,
) -> list[list[int]]:
    return [cell for row in _horizontal_cell_rows(lines, rows, gap_x, average_height) for cell in row]


def _cell_bbox(cell: list[int], boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(boxes[index][0] for index in cell),
        min(boxes[index][1] for index in cell),
        max(boxes[index][2] for index in cell),
        max(boxes[index][3] for index in cell),
    )


def _vertical_blocks(cells: list[list[int]], lines: list[OCRLine], gap_y: float, average_height: float) -> tuple[list[list[int]], list[bool]]:
    boxes = [_line_bbox(line) for line in lines]
    cell_boxes = [_cell_bbox(cell, boxes) for cell in cells]
    order = sorted(range(len(cells)), key=lambda index: (cell_boxes[index][1], cell_boxes[index][0]))
    blocks: list[list[int]] = []
    ambiguous: list[bool] = []
    # The median multiplier is the primary threshold. For sparse pages, cap
    # it by the local text height so a single large section gap cannot absorb
    # the next block merely because there are few close vertical gaps.
    threshold = min(gap_y * GAP_Y_MULTIPLIER, average_height * 0.5)
    for cell_index in order:
        current = cell_boxes[cell_index]
        candidates: list[tuple[float, int, float]] = []
        for block_index, block in enumerate(blocks):
            previous_index = block[-1]
            previous = cell_boxes[previous_index]
            same_row = abs((current[1] + current[3]) / 2 - (previous[1] + previous[3]) / 2) <= average_height * 0.8
            if same_row:
                continue
            left_distance = abs(previous[0] - current[0])
            # A wide title or paragraph can overlap several columns. The
            # stable left edge is the useful local column signal in that case.
            if left_distance > average_height * 2.5:
                continue
            gap = max(0.0, current[1] - previous[3])
            previous_text = " ".join(lines[index].text for index in cells[block[0]])
            current_text = " ".join(lines[index].text for index in cells[cell_index])
            label_value_continuation = (
                _is_known_label(previous_text)
                and not _is_known_label(current_text)
                and gap <= threshold + average_height * 0.6
            )
            if gap <= threshold or label_value_continuation:
                candidates.append((gap, block_index, gap))
        if candidates:
            _, block_index, gap = min(candidates)
            blocks[block_index].append(cell_index)
            ambiguous[block_index] = ambiguous[block_index] or abs(gap - threshold) / max(1.0, threshold) <= 0.15
        else:
            blocks.append([cell_index])
            ambiguous.append(False)
    return blocks, ambiguous


def _is_known_label(text: str) -> bool:
    words = re.findall(r"[A-Za-z]+", text.upper())
    return any(word in KNOWN_LABEL_TERMS for word in words) or text.rstrip().endswith(":") or ("(" in text and ")" in text)


def _is_wrapped_label(previous: str, current: str, average_length: float) -> bool:
    """Recognize a continuation of a label without naming a document template."""
    previous_words = previous.upper().split()
    current_words = current.upper().split()
    if not previous_words or not current_words:
        return False
    current_is_short = len(current) <= max(average_length * 1.5, 12.0)
    current_is_label_word = all(
        word.strip(".,:/()") in KNOWN_LABEL_TERMS for word in current_words
    )
    unfinished = previous_words[-1].strip(".,:/()") in {
        "OF", "BY", "FOR", "FROM", "IN", "PARTY", "AND", "OR",
    }
    return current_is_short and (current_is_label_word or unfinished)


def _split_inline_label(text: str) -> tuple[str, list[str]]:
    if ":" in text:
        label, value = text.split(":", 1)
        if label.strip() and value.strip():
            return label.strip(), [value.strip()]
        return text.rstrip(":"), []
    return text, []


def _field_from_block(
    block: list[int],
    cells: list[list[int]],
    lines: list[OCRLine],
    boxes: list[tuple[float, float, float, float]],
    method: str,
    ambiguous: bool,
    header: list[str] | None = None,
) -> OCRField:
    block_lines = [line_index for cell_index in block for line_index in cells[cell_index]]
    block_lines.sort(key=lambda index: (_line_bbox(lines[index])[1], _line_bbox(lines[index])[0]))
    raw = [lines[index].as_dict() for index in block_lines]
    first_text = " ".join(lines[index].text for index in cells[block[0]])
    label, inline_values = _split_inline_label(first_text)
    label_cell_count = 1
    average_length = statistics.mean(len(lines[index].text) for index in block_lines)
    while label_cell_count < len(block) - (1 if inline_values else 0):
        candidate = " ".join(lines[index].text for index in cells[block[label_cell_count]])
        if not _is_wrapped_label(label, candidate, average_length):
            break
        label = f"{label} {candidate}"
        label_cell_count += 1
    value_items = inline_values + [
        " ".join(lines[index].text for index in cells[cell_index])
        for cell_index in block[label_cell_count:]
    ]
    if header is not None:
        label = " | ".join(header)
        value_items = [
            " | ".join(" ".join(lines[index].text for index in cells[cell_index]) for cell_index in block)
        ]
    block_boxes = [boxes[index] for index in block_lines]
    base = 0.9 if method == "table" else 0.6
    confidence = base
    if len(block_lines) > 5:
        confidence -= 0.1
    if ambiguous:
        confidence -= 0.15
    if _is_known_label(label):
        confidence += 0.1
    return OCRField(
        label=label.rstrip(":"),
        value=value_items,
        confidence=round(max(0.0, min(1.0, confidence)), 4),
        raw_lines=raw,
        bbox={
            "x0": min(box[0] for box in block_boxes),
            "y0": min(box[1] for box in block_boxes),
            "x1": max(box[2] for box in block_boxes),
            "y1": max(box[3] for box in block_boxes),
        },
        mapping_method=method,
    )


def _repeat_signature(block: list[int], cells: list[list[int]], lines: list[OCRLine]) -> tuple[tuple[float, float], ...]:
    boxes = [_line_bbox(line) for line in lines]
    return tuple((round(_cell_bbox(cells[index], boxes)[0], -1), 0.0) for index in block)


def _similar_signature(first: tuple[tuple[float, float], ...], second: tuple[tuple[float, float], ...], tolerance: float) -> bool:
    if len(first) != len(second):
        return False
    return all(abs(left_a - left_b) <= tolerance and abs(right_a - right_b) <= tolerance for (left_a, right_a), (left_b, right_b) in zip(first, second))


def _map_page(lines: list[OCRLine]) -> tuple[list[OCRField], str]:
    if not lines:
        return [], "coordinate_fallback"
    gap_x, gap_y, average_height = _local_gaps(lines)
    rows = _rough_rows(lines, average_height)
    cell_rows = _horizontal_cell_rows(lines, rows, gap_x, average_height)
    cells = [cell for row in cell_rows for cell in row]
    cell_row_indexes: list[list[int]] = []
    offset = 0
    for row in cell_rows:
        cell_row_indexes.append(list(range(offset, offset + len(row))))
        offset += len(row)
    blocks, ambiguous = _vertical_blocks(cells, lines, gap_y, average_height)
    signatures = [_repeat_signature(block, cells, lines) for block in blocks]
    repeated = []
    for index, signature in enumerate(signatures):
        similar_count = sum(_similar_signature(signature, other, average_height) for other in signatures)
        if similar_count >= 3:
            repeated.append(index)
    repeated_rows = []
    line_boxes = [_line_bbox(line) for line in lines]
    row_signatures = [
        tuple((round(_cell_bbox(cells[cell_index], line_boxes)[0], -1), 0.0) for cell_index in row)
        for row in cell_row_indexes
    ]
    for index, signature in enumerate(row_signatures):
        similar_count = sum(_similar_signature(signature, other, average_height) for other in row_signatures)
        if similar_count >= 3:
            repeated_rows.append(index)
    repeated_run = bool(repeated_rows) and repeated_rows == list(range(repeated_rows[0], repeated_rows[-1] + 1))
    row_table = (
        len(repeated_rows) >= 3
        and repeated_run
        and len(repeated_rows) / max(1, len(cell_rows)) >= 0.5
        and any(len(cell_rows[index]) >= 2 for index in repeated_rows)
    )
    # A block signature is only promoted when the row-level clustering also
    # confirms a repeated row. This prevents wide headings and prose columns
    # from masquerading as a table on otherwise free-form pages.
    block_table = len(repeated) >= 3 and row_table and any(len(signature) >= 2 for signature in signatures)
    is_table = block_table or row_table
    fields: list[OCRField] = []
    if row_table:
        header_row = repeated_rows[0]
        header = [" ".join(lines[index].text for index in cell) for cell in cell_rows[header_row]]
        for row_index in range(len(cell_rows)):
            if row_index == header_row:
                continue
            row_cells = cell_row_indexes[row_index]
            row_ambiguous = False
            fields.append(_field_from_block(row_cells, cells, lines, [_line_bbox(line) for line in lines], "table", row_ambiguous, header=header))
    elif is_table:
        header = [" ".join(lines[index].text for index in cells[cell_index]) for cell_index in blocks[repeated[0]]]
        for block_index, block in enumerate(blocks):
            if block_index == repeated[0]:
                continue
            fields.append(_field_from_block(block, cells, lines, [_line_bbox(line) for line in lines], "table", ambiguous[block_index], header=header))
    else:
        for block_index, block in enumerate(blocks):
            fields.append(_field_from_block(block, cells, lines, [_line_bbox(line) for line in lines], "coordinate", ambiguous[block_index]))
    fields.sort(key=lambda field: (field.bbox["y0"], field.bbox["x0"]))
    return fields, "line_based" if is_table else "coordinate_fallback"


def map_ocr_lines(lines: list[OCRLine], image_path: Path | None = None) -> OCRMapping:
    del image_path
    fields: list[OCRField] = []
    modes: set[str] = set()
    for page_no in sorted({line.page_no for line in lines}):
        page_fields, mode = _map_page([line for line in lines if line.page_no == page_no])
        fields.extend(page_fields)
        modes.add(mode)
    if not fields:
        layout_mode = "coordinate_fallback"
    elif len(modes) > 1:
        layout_mode = "mixed"
    else:
        layout_mode = modes.pop()
    return OCRMapping(fields, layout_mode, sum(field.confidence < 0.6 for field in fields))


def build_field_mappings(lines: list[OCRLine]) -> list[OCRField]:
    return map_ocr_lines(lines).fields


engine = PaddleOCREngine()
