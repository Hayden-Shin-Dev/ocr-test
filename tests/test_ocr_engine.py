import unittest

from ocr_test.ocr_engine import build_field_mappings, parse_ocr_result


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

        self.assertEqual([line.text for line in result], ["first", "second"])
        self.assertEqual(result[0].confidence, 0.98)
        self.assertEqual(result[0].polygon[0], [20.0, 20.0])

    def test_ignores_empty_text(self) -> None:
        result = parse_ocr_result({"rec_texts": ["", "valid"], "rec_scores": [0.1, 0.8]})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "valid")

    def test_builds_layout_fields_from_vertical_text(self) -> None:
        lines = parse_ocr_result([{
            "rec_texts": ["CONSIGNEE (name and address)", "ACME CO., LTD.", "SEOUL, KOREA", "BOOKING NO.", "74482"],
            "rec_scores": [0.99, 0.95, 0.94, 0.98, 1.0],
            "dt_polys": [
                [[10, 10], [180, 10], [180, 24], [10, 24]],
                [[10, 28], [150, 28], [150, 43], [10, 43]],
                [[10, 47], [120, 47], [120, 62], [10, 62]],
                [[240, 10], [330, 10], [330, 24], [240, 24]],
                [[240, 28], [290, 28], [290, 43], [240, 43]],
            ],
        }])

        fields = build_field_mappings(lines)

        self.assertEqual(fields[0].label, "CONSIGNEE (name and address)")
        self.assertEqual(fields[0].value, "ACME CO., LTD. SEOUL, KOREA")
        self.assertEqual(fields[1].label, "BOOKING NO.")
        self.assertEqual(fields[1].value, "74482")


if __name__ == "__main__":
    unittest.main()
