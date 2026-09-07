import unittest

from ocr_test.ocr_engine import OCRDocument, OCRLine, build_field_mappings, map_ocr_lines, parse_ocr_result


def line(text: str, x: float, y: float, width: float = 80, height: float = 14, page_no: int = 0) -> OCRLine:
    return OCRLine(
        text=text,
        confidence=0.99,
        polygon=[[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        page_no=page_no,
    )


class OCRResultTests(unittest.TestCase):
    def test_normalizes_and_sorts_generic_lines(self) -> None:
        result = parse_ocr_result(
            [{
                "res": {
                    "rec_texts": ["second", "first"],
                    "rec_scores": [0.91, 0.98],
                    "dt_polys": [
                        [[20, 100], [80, 100], [80, 120], [20, 120]],
                        [[20, 20], [80, 20], [80, 40], [20, 40]],
                    ],
                }
            }]
        )

        self.assertEqual([item.text for item in result], ["first", "second"])
        self.assertEqual(result[0].confidence, 0.98)
        self.assertEqual(result[0].bbox, {"x0": 20.0, "y0": 20.0, "x1": 80.0, "y1": 40.0})

    def test_ignores_empty_text(self) -> None:
        result = parse_ocr_result({"rec_texts": ["", "valid"], "rec_scores": [0.1, 0.8]})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "valid")

    def test_coordinate_fallback_groups_multiline_field_without_template_rules(self) -> None:
        lines = [
            line("CONSIGNEE (name and address)", 10, 10, 170),
            line("ACME CO., LTD.", 10, 28, 140),
            line("SEOUL, KOREA", 10, 47, 110),
            line("BOOKING NO.", 240, 10, 90),
            line("74482", 240, 28, 50),
        ]

        mapping = map_ocr_lines(lines)

        self.assertEqual(mapping.layout_mode, "coordinate_fallback")
        self.assertEqual(mapping.fields[0].label, "CONSIGNEE (name and address)")
        self.assertEqual(mapping.fields[0].value, ["ACME CO., LTD.", "SEOUL, KOREA"])
        self.assertEqual(mapping.fields[1].label, "BOOKING NO.")
        self.assertEqual(mapping.fields[1].value, ["74482"])
        self.assertEqual(len(mapping.fields[0].raw_lines), 3)
        self.assertIn("bbox", mapping.fields[0].raw_lines[0])

    def test_same_row_two_columns_remain_separate_cells(self) -> None:
        lines = [
            line("BILL OF LADING", 20, 10, 210, 18),
            line("BOOKING NO.", 270, 10, 100, 18),
            line("CONSIGNEE", 20, 50, 95, 18),
            line("ACME CO.", 20, 74, 90, 18),
        ]

        fields = build_field_mappings(lines)

        self.assertEqual([field.label for field in fields], ["BILL OF LADING", "BOOKING NO.", "CONSIGNEE"])
        self.assertEqual(fields[1].value, [])
        self.assertEqual(fields[2].value, ["ACME CO."])

    def test_inline_label_value_is_kept_as_a_value_list(self) -> None:
        fields = build_field_mappings([line("PORT OF LOADING: BUSAN, KOREA", 10, 10, 250)])

        self.assertEqual(fields[0].label, "PORT OF LOADING")
        self.assertEqual(fields[0].value, ["BUSAN, KOREA"])

    def test_repeated_three_column_rows_enter_table_mode(self) -> None:
        lines = [
            line("ITEM", 10, 10, 30), line("QTY", 100, 10, 30), line("AMOUNT", 190, 10, 50),
            line("A", 10, 35, 30), line("2", 100, 35, 20), line("10", 190, 35, 25),
            line("B", 10, 60, 30), line("3", 100, 60, 20), line("15", 190, 60, 25),
        ]

        mapping = map_ocr_lines(lines)

        self.assertEqual(mapping.layout_mode, "line_based")
        self.assertEqual(len(mapping.fields), 2)
        self.assertEqual(mapping.fields[0].label, "ITEM | QTY | AMOUNT")
        self.assertEqual(mapping.fields[0].value, ["A | 2 | 10"])
        self.assertEqual(mapping.fields[0].mapping_method, "table")

    def test_wrapped_label_is_joined_before_values(self) -> None:
        fields = build_field_mappings([
            line("PLACE OF DELIVERY BY", 10, 10, 170),
            line("CARRIER", 10, 28, 70),
            line("TOKYO, JAPAN", 10, 48, 120),
        ])

        self.assertEqual(fields[0].label, "PLACE OF DELIVERY BY CARRIER")
        self.assertEqual(fields[0].value, ["TOKYO, JAPAN"])

    def test_mixed_pages_report_mixed_layout_mode(self) -> None:
        lines = [
            line("KEY", 10, 10, 30, page_no=0), line("VALUE", 100, 10, 40, page_no=0),
            line("A", 10, 35, 30, page_no=0), line("1", 100, 35, 20, page_no=0),
            line("B", 10, 60, 30, page_no=0), line("2", 100, 60, 20, page_no=0),
            line("REFERENCE:", 10, 10, 90, page_no=1), line("ABC-123", 10, 30, 70, page_no=1),
        ]

        mapping = map_ocr_lines(lines)

        self.assertEqual(mapping.layout_mode, "mixed")
        self.assertGreaterEqual(len(mapping.fields), 2)

    def test_document_response_keeps_raw_lines_and_mapping_metadata(self) -> None:
        lines = [line("REFERENCE: ABC-123", 10, 10, 150)]
        mapping = map_ocr_lines(lines)
        document = OCRDocument("sample.png", 300, 200, 12, lines, mapping.fields, mapping.layout_mode, mapping.low_confidence_count)

        payload = document.as_dict()

        self.assertEqual(payload["raw_text_lines"][0]["text"], "REFERENCE: ABC-123")
        self.assertEqual(payload["fields"][0]["value"], ["ABC-123"])
        self.assertIn(payload["layout_mode"], {"coordinate_fallback", "line_based", "mixed"})
        self.assertIn("low_confidence_count", payload)


if __name__ == "__main__":
    unittest.main()
