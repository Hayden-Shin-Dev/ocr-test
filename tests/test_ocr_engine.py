import unittest

from ocr_test.ocr_engine import parse_ocr_result


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


if __name__ == "__main__":
    unittest.main()
