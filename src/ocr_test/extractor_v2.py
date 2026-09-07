from __future__ import annotations

import os
import re
import statistics
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable, Protocol

from .field_schema import FieldDefinition, FieldSchema, load_field_schema
from .models import OCRField, OCRLine, OCRMapping
from .structure_engine import StructureDocument


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
    reading_order: list[int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticMatch:
    definition: FieldDefinition
    semantic_score: float
    exact_alias_bonus: float = 0.0


class Matcher(Protocol):
    def match_all(self, text: str, schema: FieldSchema) -> list[SemanticMatch]: ...


def _normalise(text: str) -> str:
    return re.sub(r"[^\w]+", "", text.casefold(), flags=re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"\w+", text.casefold(), flags=re.UNICODE) if token}


def _schema_lexical_score(text: str, definition: FieldDefinition) -> float:
    """A small language-independent signal for short OCR labels.

    Embeddings remain the primary semantic signal.  Short labels are often
    too terse for an embedding model, so the schema-provided name/aliases also
    contribute a soft token/character similarity.  No label is required to
    match exactly.
    """
    query = _normalise(text)
    query_tokens = _tokens(text)
    best = 0.0
    for label in definition.labels:
        candidate = _normalise(label)
        candidate_tokens = _tokens(label)
        if not candidate:
            continue
        character = SequenceMatcher(None, query, candidate).ratio()
        overlap = len(query_tokens & candidate_tokens) / max(1, min(len(query_tokens), len(candidate_tokens)))
        containment = 1.0 if query and (query in candidate or candidate in query) else 0.0
        best = max(best, 0.45 * character + 0.4 * overlap + 0.15 * containment)
    return min(1.0, best)


class MultilingualSemanticMatcher:
    """Lazy multilingual sentence embedding matcher.

    The model sees only text that has already received a structural labelness
    score. It is therefore not used as a document-wide label detector.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv(
            "OCR_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self._model: Any | None = None
        self._schema_cache: dict[tuple[str, ...], Any] = {}
        self._query_cache: dict[tuple[str, tuple[str, ...]], Any] = {}

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            try:
                import torch
                gpu_requested = os.getenv("PADDLEOCR_DEVICE", "gpu:0").startswith("gpu")
                device = "cuda" if gpu_requested and torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def match_all(self, text: str, schema: FieldSchema) -> list[SemanticMatch]:
        if not schema.fields or not text.strip():
            return []
        model = self._get_model()
        key = tuple(field.semantic_text for field in schema.fields)
        if key not in self._schema_cache:
            self._schema_cache[key] = model.encode(list(key), normalize_embeddings=True)
        query_key = (text, key)
        if query_key not in self._query_cache:
            self._query_cache[query_key] = model.encode([text], normalize_embeddings=True)[0]
        query = self._query_cache[query_key]
        return self._rank_matches(text, self._schema_cache[key] @ query, schema)

    def match_many(self, texts: list[str], schema: FieldSchema) -> dict[int, list[SemanticMatch]]:
        """Batch query embeddings so a document does not encode one region at a time."""
        if not schema.fields:
            return {index: [] for index in range(len(texts))}
        model = self._get_model()
        key = tuple(field.semantic_text for field in schema.fields)
        if key not in self._schema_cache:
            self._schema_cache[key] = model.encode(list(key), normalize_embeddings=True)
        query_indices = [index for index, text in enumerate(texts) if text.strip()]
        if not query_indices:
            return {index: [] for index in range(len(texts))}
        queries = model.encode([texts[index] for index in query_indices], normalize_embeddings=True, batch_size=32)
        return {
            index: self._rank_matches(texts[index], self._schema_cache[key] @ query, schema)
            for index, query in zip(query_indices, queries)
        }

    def _rank_matches(self, text: str, similarities: Any, schema: FieldSchema) -> list[SemanticMatch]:
        result: list[SemanticMatch] = []
        text_norm = _normalise(text)
        for definition, similarity in zip(schema.fields, similarities):
            lexical = _schema_lexical_score(text, definition)
            # Keep multilingual embeddings primary while stabilising terse
            # labels such as "Consignee" and "Invoice Number".
            semantic = 0.75 * float(similarity) + 0.25 * lexical
            bonus = 0.0
            for alias in definition.aliases:
                alias_norm = _normalise(alias)
                if alias_norm and alias_norm in text_norm:
                    bonus = max(bonus, 0.06)
            result.append(SemanticMatch(definition, semantic, bonus))
        return sorted(result, key=lambda item: item.semantic_score + item.exact_alias_bonus, reverse=True)


@dataclass(frozen=True)
class Labelness:
    score: float
    reason: str
    relations: tuple[str, ...]


@dataclass(frozen=True)
class _Candidate:
    match: SemanticMatch
    label_index: int
    selected_indices: tuple[int, ...]
    labelness: Labelness
    structure_score: float
    geometry_score: float
    datatype_score: float
    applicability_score: float
    region_index: int
    relation: tuple[str, ...]
    rank_score: float


@dataclass(frozen=True)
class ExtractorDiagnostic:
    text: str
    label_candidate: bool
    labelness_reason: str
    matched_field: str | None
    semantic_score: float
    structure_relation: list[str]
    selected_value: str | list[str] | None
    rejection_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractorV2Result:
    fields: list[OCRField]
    structure: UnifiedDocumentStructure
    layout_mode: str
    low_confidence_count: int
    schema_source: str | None
    schema_field_count: int
    diagnostics: list[ExtractorDiagnostic]

    def as_dict(self) -> dict[str, Any]:
        return {
            "fields": [field.as_dict() for field in self.fields],
            "structure": self.structure.as_dict(),
            "layout_mode": self.layout_mode,
            "low_confidence_count": self.low_confidence_count,
            "schema_source": self.schema_source,
            "schema_field_count": self.schema_field_count,
            "diagnostics": [diagnostic.as_dict() for diagnostic in self.diagnostics],
            "extractor_version": "v2-schema-structure-scored",
        }


def _area(box: dict[str, float]) -> float:
    return max(1.0, (box["x1"] - box["x0"]) * (box["y1"] - box["y0"]))


def _intersection_ratio(first: dict[str, float], second: dict[str, float]) -> float:
    x0, y0 = max(first["x0"], second["x0"]), max(first["y0"], second["y0"])
    x1, y1 = min(first["x1"], second["x1"]), min(first["y1"], second["y1"])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0) / min(_area(first), _area(second))


def _x_overlap(first: dict[str, float], second: dict[str, float]) -> float:
    overlap = max(0.0, min(first["x1"], second["x1"]) - max(first["x0"], second["x0"]))
    return overlap / max(1.0, min(first["x1"] - first["x0"], second["x1"] - second["x0"]))


def _line_indices_in_box(lines: list[OCRLine], box: dict[str, float], page_no: int) -> list[int]:
    result = []
    for index, line in enumerate(lines):
        if line.page_no != page_no:
            continue
        line_box = line.bbox
        center = ((line_box["x0"] + line_box["x1"]) / 2, (line_box["y0"] + line_box["y1"]) / 2)
        if box["x0"] <= center[0] <= box["x1"] and box["y0"] <= center[1] <= box["y1"]:
            result.append(index)
        elif _intersection_ratio(line_box, box) >= 0.25:
            result.append(index)
    return result


def build_unified_structure(lines: list[OCRLine], structure: StructureDocument | None) -> UnifiedDocumentStructure:
    if structure is None:
        return UnifiedDocumentStructure([], [], [])
    blocks = [
        UnifiedBlock(block.label, block.score, block.bbox, _line_indices_in_box(lines, block.bbox, block.page_no), block.page_no)
        for block in structure.blocks
    ]
    tables = []
    for table in structure.tables:
        cells = [
            UnifiedTableCell(cell.bbox, cell.row_index, cell.column_index, _line_indices_in_box(lines, cell.bbox, cell.page_no), cell.page_no)
            for cell in table.cells
        ]
        tables.append(UnifiedTable(table.bbox, cells, table.page_no))
    reading_order = list(structure.reading_order) or sorted(
        range(len(blocks)),
        key=lambda index: (blocks[index].page_no, blocks[index].bbox["y0"], blocks[index].bbox["x0"]),
    )
    return UnifiedDocumentStructure(blocks, tables, reading_order)


def _regions(lines: list[OCRLine], structure: UnifiedDocumentStructure) -> list[tuple[list[int], list[UnifiedTableCell], float]]:
    result: list[tuple[list[int], list[UnifiedTableCell], float]] = []
    for table in structure.tables:
        indices = _line_indices_in_box(lines, table.bbox, table.page_no)
        if indices:
            result.append((indices, table.cells, max(0.0, min(1.0, table_score(structure, table)))))
    table_boxes = [table.bbox for table in structure.tables]
    for block in structure.blocks:
        if any(_intersection_ratio(block.bbox, box) >= 0.5 for box in table_boxes):
            continue
        indices = _line_indices_in_box(lines, block.bbox, block.page_no)
        if indices:
            result.append((indices, [], max(0.0, min(1.0, block.score))))
    if not result:
        for page_no in sorted({line.page_no for line in lines}):
            indices = [index for index, line in enumerate(lines) if line.page_no == page_no]
            if indices:
                result.append((indices, [], 0.0))
    return result


def table_score(structure: UnifiedDocumentStructure, table: UnifiedTable) -> float:
    for source in structure.tables:
        if source is table:
            return 0.9
    return 0.0


def _cell_for_line(index: int, cells: list[UnifiedTableCell]) -> UnifiedTableCell | None:
    matches = [cell for cell in cells if index in cell.line_indices]
    return min(matches, key=lambda cell: _area(cell.bbox), default=None)


def _line_relation(first: OCRLine, second: OCRLine, first_cell: UnifiedTableCell | None, second_cell: UnifiedTableCell | None, local_height: float) -> tuple[float, list[str]]:
    a, b = first.bbox, second.bbox
    relations: list[str] = ["reading_order"]
    score = 0.05
    if first_cell and second_cell:
        if first_cell.row_index == second_cell.row_index:
            relations.append("same_row")
            score += 0.2
        if first_cell.column_index == second_cell.column_index:
            relations.append("same_column")
            score += 0.25
        if first_cell is second_cell:
            relations.append("same_cell")
            score += 0.25
    if _x_overlap(a, b) >= 0.15:
        score += 0.1
    if b["x0"] >= a["x1"] - local_height:
        relations.append("right")
        score += 0.12
    if b["y0"] >= a["y1"] - local_height * 0.4:
        relations.append("below")
        gap = max(0.0, b["y0"] - a["y1"])
        score += 0.18 * max(0.0, 1.0 - gap / max(local_height * 8, 1.0))
    return min(1.0, score), relations


def _labelness(index: int, region_lines: list[int], cells: list[UnifiedTableCell], lines: list[OCRLine]) -> Labelness:
    line = lines[index]
    box = line.bbox
    heights = [max(1.0, lines[item].bbox["y1"] - lines[item].bbox["y0"]) for item in region_lines]
    lengths = [len(lines[item].text) for item in region_lines]
    median_height = statistics.median(heights) if heights else 1.0
    median_length = statistics.median(lengths) if lengths else 1.0
    reasons: list[str] = []
    feature_scores: list[float] = []
    short_score = max(0.0, 1.0 - len(line.text) / max(median_length * 3.0, 1.0))
    feature_scores.append(short_score)
    if short_score > 0.45:
        reasons.append("short_region")
    has_right = any(
        other != index and abs(((lines[other].bbox["y0"] + lines[other].bbox["y1"]) / 2) - ((box["y0"] + box["y1"]) / 2)) <= median_height
        and lines[other].bbox["x0"] >= box["x1"] - median_height
        for other in region_lines
    )
    feature_scores.append(1.0 if has_right else 0.0)
    if has_right:
        reasons.append("right_value_or_column")
    has_below = any(
        other != index and lines[other].bbox["y0"] >= box["y1"] - median_height * 0.4
        and _x_overlap(box, lines[other].bbox) >= 0.15
        and lines[other].bbox["y0"] - box["y1"] <= median_height * 6
        for other in region_lines
    )
    feature_scores.append(1.0 if has_below else 0.0)
    if has_below:
        reasons.append("below_value_or_block")
    cell = _cell_for_line(index, cells)
    if cell:
        first_row = min((item.row_index for item in cells), default=cell.row_index)
        header_score = 1.0 if cell.row_index == first_row else 0.0
        feature_scores.append(header_score)
        if header_score:
            reasons.append("table_header_position")
    else:
        feature_scores.append(0.0)
    region_start = min((lines[item].bbox["y0"] for item in region_lines), default=box["y0"])
    start_score = max(0.0, 1.0 - (box["y0"] - region_start) / max(median_height * 8, 1.0))
    feature_scores.append(start_score)
    if start_score > 0.65:
        reasons.append("block_start_or_reading_start")
    inline = ":" in line.text and len(line.text.split(":", 1)[0].strip()) <= max(80, int(median_length * 2))
    feature_scores.append(1.0 if inline else 0.0)
    if inline:
        reasons.append("inline_label_value_shape")
    body_penalty = min(1.0, len(line.text) / max(median_length * 2.5, 1.0)) if len(line.text.split()) > 8 else 0.0
    if body_penalty:
        reasons.append("long_body_region")
    score = max(0.0, min(1.0, 0.12 + statistics.mean(feature_scores) * 0.88 - body_penalty * 0.65))
    reason = ",".join(reasons) if reasons else "weak_structural_labelness"
    relations = ["table_cell"] if cell else []
    return Labelness(score, reason, tuple(relations))


def _datatype_score(values: list[str], datatype: str) -> float:
    if not values:
        return 0.5
    kind = datatype.casefold()
    valid = 0
    for value in values:
        if kind in {"date", "datetime"}:
            ok = bool(re.search(r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b", value))
        elif kind in {"amount", "currency", "quantity", "weight", "measurement", "identifier"}:
            ok = bool(re.search(r"\d", value))
        else:
            ok = bool(value.strip())
        valid += int(ok)
    return valid / len(values)


def _inline_value(text: str) -> str | None:
    if ":" not in text:
        return None
    value = text.split(":", 1)[1].strip()
    return value or None


def _applicability_score(definition: FieldDefinition, lines: list[OCRLine]) -> float:
    """Use schema applicability as a soft prior without document classification."""
    if not definition.documents or not lines:
        return 0.5
    ordered = sorted(lines, key=lambda line: (line.page_no, line.bbox["y0"], line.bbox["x0"]))
    heading_window = " ".join(line.text for line in ordered[: min(5, len(ordered))]).casefold()
    if any(document.casefold() in heading_window for document in definition.documents):
        return 1.0
    return 0.5


def _candidate_values(
    candidate_index: int,
    region_lines: list[int],
    cells: list[UnifiedTableCell],
    labelness_by_line: dict[int, Labelness],
    semantic_scores: dict[int, float],
    lines: list[OCRLine],
) -> tuple[tuple[int, ...], float, tuple[str, ...]]:
    heights = [max(1.0, lines[item].bbox["y1"] - lines[item].bbox["y0"]) for item in region_lines]
    local_height = statistics.median(heights) if heights else 1.0
    first_cell = _cell_for_line(candidate_index, cells)
    ranked: list[tuple[float, int, list[str]]] = []
    first_box = lines[candidate_index].bbox
    label_score = labelness_by_line.get(candidate_index, Labelness(0.0, "", ())).score
    # A following, strong label-shaped region forms a soft boundary.  The
    # boundary is only used to decay distant value candidates; it is not a
    # label acceptance/rejection rule.
    boundary_y: float | None = None
    for index in region_lines:
        if index == candidate_index or lines[index].bbox["y0"] <= first_box["y1"]:
            continue
        current = lines[index].bbox
        if _x_overlap(first_box, current) < 0.25:
            continue
        structural = labelness_by_line.get(index, Labelness(0.0, "", ())).score
        semantic = semantic_scores.get(index, 0.0)
        boundary_score = 0.55 * structural + 0.45 * min(1.0, semantic)
        if boundary_score >= max(0.40, label_score * 0.70) and semantic >= 0.50:
            boundary_y = min(boundary_y or current["y0"], current["y0"])
    for index in region_lines:
        if index == candidate_index:
            continue
        relation_score, relation = _line_relation(lines[candidate_index], lines[index], first_cell, _cell_for_line(index, cells), local_height)
        if lines[index].bbox["y0"] < lines[candidate_index].bbox["y1"] - local_height * 0.4:
            continue
        # A line can be a value even when it is semantically close to a
        # schema field.  Labelness therefore lowers its value score softly;
        # it never removes the line from the candidate set.
        candidate_labelness = labelness_by_line.get(candidate_index, Labelness(0.0, "", ())).score
        value_labelness = labelness_by_line.get(index, Labelness(0.0, "", ())).score
        value_penalty = max(0.0, value_labelness - candidate_labelness - 0.08) * 0.35
        if len(lines[index].text.split()) > 8:
            value_penalty += 0.50
        distance_penalty = 0.0
        if boundary_y is not None and lines[index].bbox["y0"] >= boundary_y:
            distance_penalty = 0.32
        ranked.append((max(0.0, relation_score - value_penalty - distance_penalty), index, relation))
    ranked.sort(key=lambda item: (-item[0], lines[item[1]].bbox["y0"], lines[item[1]].bbox["x0"]))
    ranked = [item for item in ranked if item[0] > 0.05]
    if boundary_y is not None:
        before_boundary = [item for item in ranked if lines[item[1]].bbox["y0"] < boundary_y]
        if before_boundary:
            ranked = before_boundary
    if first_cell:
        cell_values = [item for item in ranked if "same_cell" in item[2] and "below" in item[2]]
        spatial_values = [
            item for item in ranked
            if _x_overlap(first_box, lines[item[1]].bbox) >= 0.12
            or "same_column" in item[2]
            or ("same_row" in item[2] and "right" in item[2])
        ]
        # A PP cell is the strongest value boundary.  Only when it has no
        # following text do we cross to a neighbouring row/column.
        selected = cell_values or spatial_values or ranked
    else:
        selected = [
            item for item in ranked
            if _x_overlap(first_box, lines[item[1]].bbox) >= 0.12
            or "same_row" in item[2]
        ]
    selected = sorted(selected, key=lambda item: (lines[item[1]].bbox["y0"], lines[item[1]].bbox["x0"]))[:8]
    indices = tuple(sorted([candidate_index, *(item[1] for item in selected)], key=lambda item: (lines[item].bbox["y0"], lines[item].bbox["x0"])))
    relation_names = sorted({name for _, _, relation in selected for name in relation})
    relation_scores = [item[0] for item in selected]
    relation_score = statistics.mean(relation_scores) if relation_scores else 0.0
    return indices, relation_score, tuple(relation_names)


def _field_from_candidate(candidate: _Candidate, lines: list[OCRLine]) -> OCRField:
    values: list[str] = []
    inline = _inline_value(lines[candidate.label_index].text)
    if inline:
        values.append(inline)
    values.extend(lines[index].text.strip() for index in candidate.selected_indices if index != candidate.label_index and lines[index].text.strip())
    value = [item for item in values if _datatype_score([item], candidate.match.definition.datatype) >= 0.5] or None
    boxes = [lines[index].bbox for index in candidate.selected_indices]
    return OCRField(
        label=candidate.match.definition.name,
        value=value,
        confidence=round(max(0.0, min(1.0, candidate.rank_score)), 4),
        raw_lines=[lines[index].as_dict() for index in candidate.selected_indices],
        bbox={"x0": min(box["x0"] for box in boxes), "y0": min(box["y0"] for box in boxes), "x1": max(box["x1"] for box in boxes), "y1": max(box["y1"] for box in boxes)},
        mapping_method="schema_structure_scored",
        schema_name=candidate.match.definition.name,
        datatype=candidate.match.definition.datatype,
    )


def extract_with_structure(
    lines: list[OCRLine],
    structure: StructureDocument | None = None,
    schema: FieldSchema | None = None,
    matcher: Matcher | None = None,
) -> ExtractorV2Result:
    active_schema = schema if schema is not None else load_field_schema()
    semantic_matcher = matcher or MultilingualSemanticMatcher()
    unified = build_unified_structure(lines, structure)
    regions = _regions(lines, unified)
    candidates: list[_Candidate] = []
    diagnostics: list[ExtractorDiagnostic] = []
    diagnostic_by_line: dict[int, ExtractorDiagnostic] = {}
    all_region_indices = sorted({index for region_lines, _, _ in regions for index in region_lines})
    if hasattr(semantic_matcher, "match_many"):
        batched_matches = semantic_matcher.match_many([lines[index].text for index in all_region_indices], active_schema)
        matches_by_global_index = {
            index: [match for match in batched_matches.get(position, []) if match.semantic_score + match.exact_alias_bonus >= 0.20]
            for position, index in enumerate(all_region_indices)
        }
    else:
        matches_by_global_index = {
            index: [match for match in semantic_matcher.match_all(lines[index].text, active_schema) if match.semantic_score + match.exact_alias_bonus >= 0.20]
            for index in all_region_indices
        }
    for region_index, (region_lines, cells, structure_score) in enumerate(regions):
        labelness_by_line = {index: _labelness(index, region_lines, cells, lines) for index in region_lines}
        # Labelness is a soft feature, not a gate.  Every OCR region is first
        # scored structurally; then possible label/value/schema combinations
        # are generated.  A low labelness score can still win when the other
        # semantic, geometry, structure, and datatype signals support it.
        matches_by_line = {index: matches_by_global_index.get(index, []) for index in region_lines}
        for index in region_lines:
            labelness = labelness_by_line[index]
            matches = matches_by_line.get(index, [])
            if not matches:
                rejection = "no_schema_semantic_candidate"
                diagnostic_by_line[index] = ExtractorDiagnostic(lines[index].text, labelness.score >= 0.5, labelness.reason, None, 0.0, list(labelness.relations), None, rejection)
                continue
            match = matches[0]
            semantic_scores = {item: (matches_by_line[item][0].semantic_score + matches_by_line[item][0].exact_alias_bonus) for item in region_lines if matches_by_line.get(item)}
            selected, relation_score, relation = _candidate_values(index, region_lines, cells, labelness_by_line, semantic_scores, lines)
            inline_value = _inline_value(lines[index].text)
            if len(selected) <= 1 and not inline_value:
                relation = tuple(sorted(set(relation) | ({"same_block"} if structure_score else set()) | ({"table"} if cells else set())))
                diagnostic_by_line[index] = ExtractorDiagnostic(
                    lines[index].text,
                    labelness.score >= 0.5,
                    labelness.reason,
                    match.definition.name,
                    round(match.semantic_score, 4),
                    list(relation),
                    None,
                    "no_value_relation",
                )
                continue
            values = [inline_value] if inline_value else []
            values.extend(lines[item].text for item in selected if item != index)
            values = [value for value in values if value]
            datatype_score = _datatype_score(values, match.definition.datatype)
            geometry_score = min(1.0, relation_score + (0.15 if "same_cell" in relation else 0.0))
            applicability_score = _applicability_score(match.definition, lines)
            relation = tuple(sorted(set(relation) | ({"same_block"} if structure_score else set()) | ({"table"} if cells else set())))
            rank = min(1.0, 0.56 * match.semantic_score + 0.14 * labelness.score + 0.12 * relation_score + 0.06 * geometry_score + 0.08 * datatype_score + 0.04 * applicability_score + match.exact_alias_bonus)
            candidates.append(_Candidate(match, index, selected, labelness, structure_score, geometry_score, datatype_score, applicability_score, region_index, relation, rank))
            diagnostic_by_line[index] = ExtractorDiagnostic(lines[index].text, labelness.score >= 0.5, labelness.reason, match.definition.name, round(match.semantic_score, 4), list(relation), None, None)

    candidates.sort(key=lambda candidate: candidate.rank_score, reverse=True)
    chosen: dict[str, OCRField] = {}
    chosen_candidates: dict[str, _Candidate] = {}
    used: set[int] = set()
    for candidate in candidates:
        name = candidate.match.definition.name
        if name in chosen or candidate.rank_score < 0.55:
            continue
        if any(index in used for index in candidate.selected_indices):
            continue
        chosen[name] = _field_from_candidate(candidate, lines)
        chosen_candidates[name] = candidate
        used.update(candidate.selected_indices)

    for index, diagnostic in list(diagnostic_by_line.items()):
        selected_field = chosen_candidates.get(diagnostic.matched_field or "")
        if selected_field and selected_field.label_index == index:
            field = chosen[selected_field.match.definition.name]
            diagnostic_by_line[index] = ExtractorDiagnostic(diagnostic.text, diagnostic.label_candidate, diagnostic.labelness_reason, diagnostic.matched_field, diagnostic.semantic_score, list(selected_field.relation), field.value, None)
        elif diagnostic.matched_field and diagnostic.rejection_reason is None:
            diagnostic_by_line[index] = ExtractorDiagnostic(diagnostic.text, diagnostic.label_candidate, diagnostic.labelness_reason, diagnostic.matched_field, diagnostic.semantic_score, diagnostic.structure_relation, None, "lower_rank_or_evidence_conflict")
    diagnostics = [diagnostic_by_line[index] for index in sorted(diagnostic_by_line)]

    empty = {"x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}
    fields = [chosen.get(definition.name, OCRField(definition.name, None, 0.0, [], empty, "unresolved", definition.name, definition.datatype)) for definition in active_schema.fields]
    mode = "pp_structure" if unified.blocks or unified.tables else "coordinate_fallback"
    return ExtractorV2Result(fields, unified, mode, sum(field.confidence < 0.6 for field in fields if field.value is not None), active_schema.source, len(active_schema.fields), diagnostics)


def map_ocr_lines(lines: list[OCRLine], schema: FieldSchema | None = None, matcher: Matcher | None = None) -> OCRMapping:
    result = extract_with_structure(lines, None, schema=schema, matcher=matcher)
    return OCRMapping(result.fields, result.layout_mode, result.low_confidence_count)
