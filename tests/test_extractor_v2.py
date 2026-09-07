import unittest

from ocr_test.extractor_v2 import SemanticMatch, _schema_lexical_score, extract_with_structure
from ocr_test.field_schema import FieldSchema, load_field_schema
from ocr_test.models import OCRLine


def line(text: str, x: float, y: float, width: float = 120, height: float = 18) -> OCRLine:
    return OCRLine(text, 0.99, [[x, y], [x + width, y], [x + width, y + height], [x, y + height]])


class SchemaMatcher:
    """Deterministic test matcher; production uses multilingual embeddings."""

    def match_all(self, text: str, schema: FieldSchema) -> list[SemanticMatch]:
        if text.startswith("The receiver"):
            return sorted(
                (SemanticMatch(field, 0.95 if field.name == "payment_terms" else 0.05) for field in schema.fields),
                key=lambda match: match.semantic_score,
                reverse=True,
            )
        return sorted(
            (SemanticMatch(field, _schema_lexical_score(text, field)) for field in schema.fields),
            key=lambda match: match.semantic_score,
            reverse=True,
        )


class ExtractorV2Tests(unittest.TestCase):
    def test_body_text_is_not_promoted_to_label_without_value_relation(self) -> None:
        schema = FieldSchema.from_payload([
            {"name": "consignee", "datatype": "party", "description": "Party receiving the shipped goods."},
            {"name": "payment_terms", "datatype": "text", "description": "Conditions defining payment."},
        ], source="test")
        lines = [
            line("Consignee", 0, 0),
            line("U and E Dominion Strategies Co., Ltd.", 0, 25, 300),
            line("128 Main Street, Seoul", 0, 50, 200),
            line("The receiver must inspect the cargo after arrival at the discharge port", 0, 100, 600),
        ]

        result = extract_with_structure(lines, schema=schema, matcher=SchemaMatcher())

        consignee = next(field for field in result.fields if field.label == "consignee")
        payment = next(field for field in result.fields if field.label == "payment_terms")
        body_diagnostic = next(item for item in result.diagnostics if item.text.startswith("The receiver"))
        self.assertEqual(consignee.value, ["U and E Dominion Strategies Co., Ltd.", "128 Main Street, Seoul"])
        self.assertIsNone(payment.value)
        self.assertFalse(body_diagnostic.label_candidate)
        self.assertEqual(body_diagnostic.rejection_reason, "no_value_relation")

    def test_existing_schema_is_loaded_without_creating_fields(self) -> None:
        schema = load_field_schema()
        self.assertEqual(len(schema.fields), 72)
        self.assertIn("consignee", schema.by_name)
        self.assertIn("receiver", schema.by_name["consignee"].aliases)


if __name__ == "__main__":
    unittest.main()
