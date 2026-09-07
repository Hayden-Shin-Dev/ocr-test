from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class FieldDefinition:
    name: str
    aliases: tuple[str, ...] = ()
    datatype: str = "text"
    description: str = ""
    scope: str = "document"
    documents: tuple[str, ...] = ()

    @property
    def labels(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    @property
    def semantic_text(self) -> str:
        human_name = self.name.replace("_", " ")
        return " ".join(part for part in (human_name, *self.aliases, self.description, self.datatype) if part)


@dataclass(frozen=True)
class FieldSchema:
    fields: tuple[FieldDefinition, ...]
    source: str | None = None
    name: str | None = None
    version: str | None = None

    @property
    def by_name(self) -> dict[str, FieldDefinition]:
        return {field.name: field for field in self.fields}

    @classmethod
    def from_payload(cls, payload: Any, source: str | None = None) -> "FieldSchema":
        metadata = payload if isinstance(payload, Mapping) else {}
        if isinstance(payload, Mapping):
            payload = payload.get("fields", payload.get("schema", []))
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise ValueError("field schema must contain a fields list")
        fields: list[FieldDefinition] = []
        for item in payload:
            if isinstance(item, str):
                fields.append(FieldDefinition(item))
                continue
            if not isinstance(item, Mapping) or not str(item.get("name", "")).strip():
                continue
            aliases = item.get("aliases", ())
            if isinstance(aliases, str):
                aliases = (aliases,)
            if not isinstance(aliases, Sequence):
                aliases = ()
            fields.append(FieldDefinition(
                name=str(item["name"]).strip(),
                aliases=tuple(str(alias).strip() for alias in aliases if str(alias).strip()),
                datatype=str(item.get("datatype", item.get("kind", "text"))),
                description=str(item.get("description", "")).strip(),
                scope=str(item.get("scope", "document")),
                documents=tuple(str(document).strip() for document in item.get("documents", ()) if str(document).strip()),
            ))
        return cls(tuple(fields), source, metadata.get("schema_name"), metadata.get("version"))


def _candidate_schema_paths() -> list[Path]:
    project_root = Path(__file__).resolve().parents[2]
    configured = os.getenv("OCR_FIELD_SCHEMA")
    paths = [Path(configured)] if configured else []
    paths.extend([
        Path.cwd() / "audit_schema_v2.json",
        project_root / "config" / "audit_field_schema_v2.json",
        project_root / "data" / "audit_field_schema_v2.json",
        project_root / "audit_field_schema_v2.json",
    ])
    return paths


def load_field_schema() -> FieldSchema:
    for path in _candidate_schema_paths():
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig") as handle:
            return FieldSchema.from_payload(json.load(handle), str(path))
    return FieldSchema((), None)
