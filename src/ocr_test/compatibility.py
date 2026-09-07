from __future__ import annotations

import statistics
import re

from .field_schema import FieldDefinition, FieldSchema
from .extractor_v2 import SemanticMatch, extract_with_structure
from .models import OCRField, OCRLine, OCRMapping


def _looks_like_legacy_label(text: str, average_length: float) -> bool:
    value = text.strip()
    if not value:
        return False
    if ":" in value or "(" in value and ")" in value:
        return True
    words = [word for word in value.split() if any(character.isalpha() for character in word)]
    uppercase_words = [word for word in words if word.upper() == word]
    return bool(words and len(uppercase_words) / len(words) >= 0.7 and len(value) <= max(average_length * 1.8, 80.0))


def _compatibility_schema(lines: list[OCRLine]) -> FieldSchema:
    """Build only the legacy response envelope's transient schema.

    This is not used by Extractor v2. It exists solely because the old
    /api/ocr endpoint accepted arbitrary labels before Audit Schema v2 was
    added. No company/value/document vocabulary is embedded here.
    """
    if not lines:
        return FieldSchema((), "compatibility")
    average_length = statistics.mean(len(line.text) for line in lines)
    definitions: list[FieldDefinition] = []
    seen: set[str] = set()
    selected: list[OCRLine] = []
    for line in lines:
        if not _looks_like_legacy_label(line.text, average_length):
            continue
        box = line.bbox
        continuation = False
        for previous in selected:
            previous_box = previous.bbox
            overlap = max(0.0, min(box["x1"], previous_box["x1"]) - max(box["x0"], previous_box["x0"]))
            overlap /= max(1.0, min(box["x1"] - box["x0"], previous_box["x1"] - previous_box["x0"]))
            gap = box["y0"] - previous_box["y1"]
            if overlap >= 0.5 and 0 <= gap <= max(average_length, 40.0):
                continuation = True
                break
        if continuation:
            continue
        name = line.text.split(":", 1)[0].strip()
        key = name.casefold()
        if key in seen:
            continue
        definitions.append(FieldDefinition(name=name, aliases=(line.text,), datatype="text"))
        seen.add(key)
        selected.append(line)
    return FieldSchema(tuple(definitions), "compatibility")


def map_ocr_lines(lines: list[OCRLine]) -> OCRMapping:
    result = extract_with_structure(lines, None, schema=_compatibility_schema(lines), matcher=_CompatibilityMatcher())
    return OCRMapping(result.fields, result.layout_mode, result.low_confidence_count)


def build_field_mappings(lines: list[OCRLine]) -> list[OCRField]:
    return map_ocr_lines(lines).fields


class _CompatibilityMatcher:
    def match_all(self, text: str, schema: FieldSchema) -> list[SemanticMatch]:
        normalized = re.sub(r"[^\w]+", "", text.casefold(), flags=re.UNICODE)
        result = []
        for definition in schema.fields:
            aliases = (*definition.labels, definition.name)
            score = max((1.0 for alias in aliases if re.sub(r"[^\w]+", "", alias.casefold()) == normalized), default=0.0)
            if score:
                result.append(SemanticMatch(definition, score))
        return result
