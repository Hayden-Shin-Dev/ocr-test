from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .ocr_engine import OCRField, OCRLine, OCRMapping, _line_bbox, map_ocr_lines
from .structure_engine import StructureBlock, StructureCell, StructureDocument, StructureTable


LABEL_WORDS = {
    "ADDRESS", "AMOUNT", "BILL", "BOOKING", "CARRIER", "CONSIGNEE", "COUNTRY",
    "DATE", "DESCRIPTION", "DESTINATION", "DELIVERY", "DISPATCH", "EXPORT",
    "FINAL", "FORWARDING", "FREIGHT", "HS", "INVOICE", "LETTER", "LOADING",
    "MARINE", "METHOD", "NAME", "NUMBER", "NO", "NOTIFY", "ORIGIN", "PACKAGES",
    "PARTY", "PAYMENT", "PORT", "PRICE", "PRODUCT", "QUANTITY", "REFERENCE",
    "SHIPMENT", "SHIPPER", "SIGNATURE", "SIGNATORY", "TERMS", "TOTAL", "TYPE",
    "UNIT", "VALUE", "VESSEL", "VOYAGE", "WEIGHT", "ZIP", "CODE", "CURRENCY",
}


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
class ExtractorV2Result:
    fields: list[OCRField]
    structure: UnifiedDocumentStructure
    layout_mode: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "fields": [field.as_dict() for field in self.fields],
            "structure": self.structure.as_dict(),
            "layout_mode": self.layout_mode,
            "extractor_version": "v2-pp-structure",
        }


def _area(box: dict[str, float]) -> float:
    return max(1.0, (box["x1"] - box["x0"]) * (box["y1"] - box["y0"]))


def _intersection_ratio(first: dict[str, float], second: dict[str, float]) -> float:
    x0 = max(first["x0"], second["x0"])
    y0 = max(first["y0"], second["y0"])
    x1 = min(first["x1"], second["x1"])
    y1 = min(first["y1"], second["y1"])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0) / min(_area(first), _area(second))


def _line_center(line: OCRLine) -> tuple[float, float]:
    box = _line_bbox(line)
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _line_indices_in_box(lines: list[OCRLine], box: dict[str, float], page_no: int) -> list[int]:
    result: list[int] = []
    for index, line in enumerate(lines):
        if line.page_no != page_no:
            continue
        line_box = line.bbox
        center_x, center_y = _line_center(line)
        if (
            box["x0"] <= center_x <= box["x1"]
            and box["y0"] <= center_y <= box["y1"]
        ) or _intersection_ratio(line_box, box) >= 0.25:
            result.append(index)
    return result


def build_unified_structure(lines: list[OCRLine], structure: StructureDocument) -> UnifiedDocumentStructure:
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
    tables: list[UnifiedTable] = []
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


def _rows(lines: list[OCRLine], indices: Iterable[int]) -> list[list[int]]:
    selected = list(indices)
    if not selected:
        return []
    heights = [max(1.0, _line_bbox(lines[index])[3] - _line_bbox(lines[index])[1]) for index in selected]
    tolerance = statistics.median(heights) * 0.6
    rows: list[list[int]] = []
    for index in sorted(selected, key=lambda item: (_line_center(lines[item])[1], _line_center(lines[item])[0])):
        center_y = _line_center(lines[index])[1]
        matching = [row for row in rows if abs(statistics.mean(_line_center(lines[item])[1] for item in row) - center_y) <= tolerance]
        if matching:
            matching[0].append(index)
        else:
            rows.append([index])
    for row in rows:
        row.sort(key=lambda item: _line_center(lines[item])[0])
    return rows


def _join_rows(lines: list[OCRLine], rows: list[list[int]]) -> list[str]:
    return [" ".join(lines[index].text for index in row) for row in rows]


def _is_label(text: str) -> bool:
    words = [word.strip(".,:/()") for word in text.upper().split()]
    return bool(text.rstrip().endswith(":") or any(word in LABEL_WORDS for word in words))


def _split_inline(text: str) -> tuple[str, list[str]]:
    if ":" not in text:
        return text, []
    label, value = text.split(":", 1)
    return (label.strip(), [value.strip()]) if label.strip() and value.strip() else (text.rstrip(":"), [])


def _field_from_lines(lines: list[OCRLine], indices: list[int], bbox: dict[str, float], score: float) -> OCRField | None:
    if not indices:
        return None
    grouped = _join_rows(lines, _rows(lines, indices))
    if not grouped:
        return None
    label, inline_values = _split_inline(grouped[0])
    values = inline_values + grouped[1:]
    average_ocr = statistics.mean(lines[index].confidence for index in indices)
    confidence = min(1.0, 0.55 + score * 0.2 + average_ocr * 0.25)
    return OCRField(
        label=label.rstrip(":"),
        value=values,
        confidence=round(confidence, 4),
        raw_lines=[lines[index].as_dict() for index in indices],
        bbox=bbox,
        mapping_method="pp_structure",
    )


def _table_has_header_row(table: UnifiedTable, lines: list[OCRLine]) -> bool:
    row_groups: dict[int, list[UnifiedTableCell]] = {}
    for cell in table.cells:
        row_groups.setdefault(cell.row_index, []).append(cell)
    if not row_groups:
        return False
    first_row = row_groups[min(row_groups)]
    populated = [cell for cell in first_row if cell.line_indices]
    if len(populated) < 4:
        return False
    return sum(_is_label(" ".join(lines[index].text for index in cell.line_indices)) for cell in populated) >= len(populated) * 0.5


def _table_fields(table: UnifiedTable, lines: list[OCRLine]) -> list[OCRField]:
    if not table.cells:
        return []
    by_row: dict[int, list[UnifiedTableCell]] = {}
    by_column: dict[int, list[UnifiedTableCell]] = {}
    for cell in table.cells:
        by_row.setdefault(cell.row_index, []).append(cell)
        by_column.setdefault(cell.column_index, []).append(cell)
    fields: list[OCRField] = []
    if _table_has_header_row(table, lines):
        header_row = min(by_row)
        headers = {cell.column_index: " ".join(lines[index].text for index in cell.line_indices) for cell in by_row[header_row] if cell.line_indices}
        for column, header in headers.items():
            data_cells = [cell for cell in by_column.get(column, []) if cell.row_index != header_row and cell.line_indices]
            if not data_cells:
                continue
            values = [" ".join(lines[index].text for index in _rows(lines, cell.line_indices)[0:1][0]) if _rows(lines, cell.line_indices) else "" for cell in data_cells]
            indices = [index for cell in data_cells for index in cell.line_indices]
            bbox = {
                "x0": min(cell.bbox["x0"] for cell in data_cells),
                "y0": min(cell.bbox["y0"] for cell in data_cells),
                "x1": max(cell.bbox["x1"] for cell in data_cells),
                "y1": max(cell.bbox["y1"] for cell in data_cells),
            }
            fields.append(OCRField(
                label=header,
                value=[value for value in values if value],
                confidence=0.85,
                raw_lines=[lines[index].as_dict() for index in indices],
                bbox=bbox,
                mapping_method="pp_structure_table",
            ))
        return fields
    for cell in sorted(table.cells, key=lambda item: (item.row_index, item.column_index)):
        field = _field_from_lines(lines, cell.line_indices, cell.bbox, 0.8)
        if field:
            fields.append(field)
    return fields


def extract_with_structure(lines: list[OCRLine], structure: StructureDocument) -> ExtractorV2Result:
    unified = build_unified_structure(lines, structure)
    fields: list[OCRField] = []
    covered: set[int] = set()
    for table in unified.tables:
        fields.extend(_table_fields(table, lines))
        covered.update(index for cell in table.cells for index in cell.line_indices)
    for block in unified.blocks:
        scoped = [index for index in block.line_indices if index not in covered]
        if not scoped or block.label.lower() in {"table", "figure", "image"}:
            continue
        field = _field_from_lines(lines, scoped, block.bbox, block.score)
        if field:
            fields.append(field)
            covered.update(scoped)
    if not fields:
        baseline: OCRMapping = map_ocr_lines(lines)
        fields = baseline.fields
        mode = baseline.layout_mode
    else:
        mode = "pp_structure"
    fields.sort(key=lambda field: (field.bbox["y0"], field.bbox["x0"]))
    return ExtractorV2Result(fields, unified, mode)
