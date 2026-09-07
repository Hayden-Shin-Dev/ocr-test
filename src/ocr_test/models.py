from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float
    polygon: list[list[float]]
    page_no: int = 0

    @property
    def bbox(self) -> dict[str, float]:
        points = self.polygon
        xs = [point[0] for point in points if len(point) >= 2]
        ys = [point[1] for point in points if len(point) >= 2]
        return {
            "x0": min(xs, default=0.0),
            "y0": min(ys, default=0.0),
            "x1": max(xs, default=0.0),
            "y1": max(ys, default=0.0),
        }

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
    value: str | list[str] | None
    confidence: float
    raw_lines: list[dict[str, Any]]
    bbox: dict[str, float]
    mapping_method: str
    schema_name: str | None = None
    datatype: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OCRMapping:
    fields: list[OCRField]
    layout_mode: str
    low_confidence_count: int


@dataclass(frozen=True)
class OCRDocument:
    """OCR output plus optional compatibility fields.

    PaddleOCREngine only fills source metadata and ``lines``. The optional
    fields exist so the legacy API envelope can remain backwards compatible
    after a compatibility mapper is applied in the application layer.
    """

    source_name: str
    width: int
    height: int
    elapsed_ms: int
    lines: list[OCRLine]
    fields: list[OCRField] = field(default_factory=list)
    layout_mode: str = "raw"
    low_confidence_count: int = 0

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def structured_text(self) -> str:
        def format_value(value: str | list[str] | None) -> str:
            if value is None:
                return ""
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
