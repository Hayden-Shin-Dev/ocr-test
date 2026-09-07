from __future__ import annotations

import difflib
import re
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .field_schema import FieldDefinition, FieldSchema, load_field_schema
from .models import OCRField, OCRLine, OCRMapping
from .structure_engine import StructureBlock, StructureCell, StructureDocument, StructureTable


@dataclass(frozen=True)
class UnifiedBlock:
    label: str
    score: float
    bbox: dict[str, float]
    line_indices: list[int]
    page_no: int = 0


@dataclass(frozen=True)
class UnifiedTableCell:
    bbox: dict[str, float]
    row_index: int
    column_index: int
    line_indices: list[int]
    page_no: int = 0


@dataclass(frozen=True)
class UnifiedTable:
    bbox: dict[str, float]
    cells: list[UnifiedTableCell]
    page_no: int = 0


@dataclass(frozen=True)
class UnifiedDocumentStructure:
    blocks: list[UnifiedBlock]
    tables: list[UnifiedTable]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _LabelCandidate:
    definition: FieldDefinition
    line_index: int
    label_text: str
    inline_value: str | None
    match_score: float
    region_index: int
    cell: UnifiedTableCell | None = None


@dataclass(frozen=True)
class ExtractorV2Result:
    fields: list[OCRField]
    structure: UnifiedDocumentStructure
    layout_mode: str
    low_confidence_count: int
    schema_source: str | None
    schema_field_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "fields": [field.as_dict() for field in self.fields],
            "structure": self.structure.as_dict(),
            "layout_mode": self.layout_mode,
            "low_confidence_count": self.low_confidence_count,
            "schema_source": self.schema_source,
            "schema_field_count": self.schema_field_count,
            "extractor_version": "v2-schema-driven",
        }


def _box(line: OCRLine) -> dict[str, float]:
    return line.bbox


def _area(box: dict[str, float]) -> float:
    return max(1.0, (box["x1"] - box["x0"]) * (box["y1"] - box["y0"]))


def _intersection_ratio(first: dict[str, float], second: dict[str, float]) -> float:
    x0 = max(first["x0"], second["x0"])
    y0 = max(first["y0"], second["y0"])
    x1 = min(first["x1"], second["x1"])
    y1 = min(first["y1"], second["y1"])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return intersection / min(_area(first), _area(second))


def _x_overlap(first: dict[str, float], second: dict[str, float]) -> float:
    overlap = max(0.0, min(first["x1"], second["x1"]) - max(first["x0"], second["x0"]))
    return overlap / max(1.0, min(first["x1"] - first["x0"], second["x1"] - second["x0"]))


def _line_indices_in_box(lines: list[OCRLine], box: dict[str, float], page_no: int) -> list[int]:
    result: list[int] = []
    for index, line in enumerate(lines):
        if line.page_no != page_no:
            continue
        line_box = _box(line)
        center_x = (line_box["x0"] + line_box["x1"]) / 2
        center_y = (line_box["y0"] + line_box["y1"]) / 2
        if (
            box["x0"] <= center_x <= box["x1"]
            and box["y0"] <= center_y <= box["y1"]
        ) or _intersection_ratio(line_box, box) >= 0.25:
            result.append(index)
    return result


def build_unified_structure(lines: list[OCRLine], structure: StructureDocument | None) -> UnifiedDocumentStructure:
    if structure is None:
        return UnifiedDocumentStructure([], [])
    blocks = [
        UnifiedBlock(
            label=block.label,
            score=block.score,
            bbox=block.bbox,
            line_indices=_line_indices_in_box(lines, block.bbox, block.page_no),
            page_no=block.page_no,
        )
        for block in structure.blocks
    ]
    tables = []
    for table in structure.tables:
        cells = [
            UnifiedTableCell(
                bbox=cell.bbox,
                row_index=cell.row_index,
                column_index=cell.column_index,
                line_indices=_line_indices_in_box(lines, cell.bbox, cell.page_no),
                page_no=cell.page_no,
            )
            for cell in table.cells
        ]
        tables.append(UnifiedTable(table.bbox, cells, table.page_no))
    return UnifiedDocumentStructure(blocks, tables)


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _label_match(text: str, definition: FieldDefinition) -> tuple[float, str, str | None]:
    source = text.strip()
    left, separator, right = source.partition(":")
    candidates = [(alias, left if separator else source) for alias in definition.labels if alias.strip()]
    best_score = 0.0
    best_label = source
    best_value: str | None = None
    for alias, comparison in candidates:
        alias_norm = _normalise(alias)
        comparison_norm = _normalise(comparison)
        if not alias_norm or not comparison_norm:
            continue
        if comparison_norm == alias_norm:
            score = 1.0
        elif comparison_norm.startswith(alias_norm):
            score = 0.91
        elif alias_norm in comparison_norm and len(alias_norm) >= 5:
            score = 0.84
        else:
            score = difflib.SequenceMatcher(None, comparison_norm, alias_norm).ratio()
            if score < 0.78:
                continue
        if score > best_score:
            best_score = score
            best_label = left.strip() if separator else source
            best_value = right.strip() if separator and right.strip() else None
    return best_score, best_label, best_value


def _best_label(text: str, schema: FieldSchema) -> tuple[FieldDefinition, float, str, str | None] | None:
    matches = []
    for definition in schema.fields:
        score, label, value = _label_match(text, definition)
        if score:
            matches.append((score, definition, label, value))
    if not matches:
        return None
    score, definition, label, value = max(matches, key=lambda item: item[0])
    return definition, score, label, value


def _region_units(lines: list[OCRLine], unified: UnifiedDocumentStructure) -> list[tuple[list[int], list[UnifiedTableCell], float]]:
    units: list[tuple[list[int], list[UnifiedTableCell], float]] = []
    for table in unified.tables:
        indices = _line_indices_in_box(lines, table.bbox, table.page_no)
        if indices:
            units.append((indices, table.cells, 0.9))
    table_boxes = [table.bbox for table in unified.tables]
    for block in unified.blocks:
        if block.label.casefold() in {"table", "figure", "image"}:
            continue
        if any(_intersection_ratio(block.bbox, table_box) >= 0.5 for table_box in table_boxes):
            continue
        if block.line_indices:
            units.append((block.line_indices, [], max(0.0, min(1.0, block.score))))
    if not units:
        for page_no in sorted({line.page_no for line in lines}):
            indices = [index for index, line in enumerate(lines) if line.page_no == page_no]
            if indices:
                units.append((indices, [], 0.0))
    return units


def _cell_for_line(index: int, cells: list[UnifiedTableCell]) -> UnifiedTableCell | None:
    matches = [cell for cell in cells if index in cell.line_indices]
    return min(matches, key=lambda cell: _area(cell.bbox), default=None)


def _candidate_lines(
    candidate: _LabelCandidate,
    region_lines: list[int],
    label_candidates: list[_LabelCandidate],
    lines: list[OCRLine],
    cells: list[UnifiedTableCell],
) -> list[int]:
    label_box = _box(lines[candidate.line_index])
    heights = [max(1.0, _box(lines[index])["y1"] - _box(lines[index])["y0"]) for index in region_lines]
    local_height = statistics.median(heights) if heights else 1.0
    next_boundary: float | None = None
    for other in label_candidates:
        if other.line_index == candidate.line_index:
            continue
        other_box = _box(lines[other.line_index])
        if other_box["y0"] <= label_box["y0"] + local_height:
            continue
        if _x_overlap(label_box, other_box) >= 0.15:
            next_boundary = other_box["y0"]
            break

    selected = [candidate.line_index]
    candidate_cell = candidate.cell
    for index in sorted(region_lines, key=lambda item: (_box(lines[item])["y0"], _box(lines[item])["x0"])):
        if index == candidate.line_index:
            continue
        current_box = _box(lines[index])
        if current_box["y0"] < label_box["y1"] - local_height * 0.35:
            continue
        if next_boundary is not None and current_box["y0"] >= next_boundary:
            continue
        current_cell = _cell_for_line(index, cells)
        same_column = bool(candidate_cell and current_cell and candidate_cell.column_index == current_cell.column_index)
        same_row = bool(candidate_cell and current_cell and candidate_cell.row_index == current_cell.row_index)
        overlap = _x_overlap(label_box, current_box)
        nearby_column = abs(((label_box["x0"] + label_box["x1"]) / 2) - ((current_box["x0"] + current_box["x1"]) / 2)) <= local_height * 4
        if overlap >= 0.12 or (same_column and current_box["y0"] > label_box["y1"]) or (same_row and current_box["x0"] >= label_box["x1"]):
            selected.append(index)
        elif nearby_column and current_box["y0"] - label_box["y1"] <= local_height * 2:
            selected.append(index)
    return sorted(set(selected), key=lambda item: (_box(lines[item])["y0"], _box(lines[item])["x0"]))


def _datatype_valid(value: str, datatype: str) -> bool:
    kind = datatype.casefold()
    if not value.strip():
        return False
    if kind in {"number", "integer", "decimal", "quantity", "amount"}:
        return bool(re.search(r"\d", value))
    if kind in {"date", "datetime"}:
        return bool(re.search(r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b", value))
    if kind in {"currency", "money"}:
        return bool(re.search(r"(?:[$€£¥]|\b[A-Z]{3}\b)\s*[-+]?\d", value, re.I))
    return True


def _union_bbox(lines: list[OCRLine], indices: Iterable[int]) -> dict[str, float]:
    boxes = [_box(lines[index]) for index in indices]
    return {
        "x0": min(box["x0"] for box in boxes),
        "y0": min(box["y0"] for box in boxes),
        "x1": max(box["x1"] for box in boxes),
        "y1": max(box["y1"] for box in boxes),
    }


def _make_field(
    candidate: _LabelCandidate,
    selected: list[int],
    lines: list[OCRLine],
    structure_score: float,
) -> OCRField:
    values: list[str] = []
    if candidate.inline_value:
        values.append(candidate.inline_value)
    for index in selected:
        if index == candidate.line_index:
            continue
        text = lines[index].text.strip()
        if text:
            values.append(text)
    valid_values = [value for value in values if _datatype_valid(value, candidate.definition.datatype)]
    average_ocr = statistics.mean(lines[index].confidence for index in selected)
    datatype_score = 1.0 if not values or len(valid_values) == len(values) else 0.65
    confidence = min(1.0, 0.45 * candidate.match_score + 0.25 * average_ocr + 0.2 * structure_score + 0.1 * datatype_score)
    return OCRField(
        label=candidate.definition.name,
        value=valid_values or None,
        confidence=round(confidence, 4),
        raw_lines=[lines[index].as_dict() for index in selected],
        bbox=_union_bbox(lines, selected),
        mapping_method="schema_geometry" if structure_score else "schema_coordinate",
        schema_name=candidate.definition.name,
        datatype=candidate.definition.datatype,
    )


def extract_with_structure(
    lines: list[OCRLine],
    structure: StructureDocument | None = None,
    schema: FieldSchema | None = None,
) -> ExtractorV2Result:
    """Resolve OCR evidence against the externally supplied field schema."""
    active_schema = schema if schema is not None else load_field_schema()
    unified = build_unified_structure(lines, structure)
    regions = _region_units(lines, unified)
    candidates: list[tuple[_LabelCandidate, list[int], float]] = []
    for region_index, (region_lines, cells, structure_score) in enumerate(regions):
        labels: list[_LabelCandidate] = []
        for index in region_lines:
            match = _best_label(lines[index].text, active_schema)
            if not match:
                continue
            definition, score, label, inline_value = match
            labels.append(_LabelCandidate(definition, index, label, inline_value, score, region_index, _cell_for_line(index, cells)))
        for candidate in labels:
            selected = _candidate_lines(candidate, region_lines, labels, lines, cells)
            candidates.append((candidate, selected, structure_score))

    # One schema role and one OCR evidence span may be assigned only once.
    candidates.sort(key=lambda item: (item[0].match_score, len(item[1])), reverse=True)
    chosen_by_name: dict[str, OCRField] = {}
    used_lines: set[int] = set()
    for candidate, selected, structure_score in candidates:
        name = candidate.definition.name
        if name in chosen_by_name or any(index in used_lines for index in selected):
            continue
        field = _make_field(candidate, selected, lines, structure_score)
        chosen_by_name[name] = field
        used_lines.update(selected)

    fields: list[OCRField] = []
    empty_bbox = {"x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}
    for definition in active_schema.fields:
        fields.append(chosen_by_name.get(definition.name, OCRField(
            label=definition.name,
            value=None,
            confidence=0.0,
            raw_lines=[],
            bbox=empty_bbox,
            mapping_method="unresolved",
            schema_name=definition.name,
            datatype=definition.datatype,
        )))
    fields.sort(key=lambda field: (field.bbox["y0"], field.bbox["x0"], field.label))
    mode = "pp_structure" if unified.blocks or unified.tables else "coordinate_fallback"
    return ExtractorV2Result(
        fields=fields,
        structure=unified,
        layout_mode=mode,
        low_confidence_count=sum(field.confidence < 0.6 for field in fields if field.value is not None),
        schema_source=active_schema.source,
        schema_field_count=len(active_schema.fields),
    )


def map_ocr_lines(lines: list[OCRLine], schema: FieldSchema | None = None) -> OCRMapping:
    """Extractor-owned mapping entry point used by compatibility code."""
    result = extract_with_structure(lines, None, schema=schema)
    return OCRMapping(result.fields, result.layout_mode, result.low_confidence_count)
