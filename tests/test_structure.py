import unittest

from ocr_test.extractor_v2 import extract_with_structure
from ocr_test.field_schema import FieldDefinition, FieldSchema
from ocr_test.ocr_engine import OCRLine
from ocr_test.structure_engine import parse_structure_result


def text_line(text: str, x: float, y: float, width: float = 20, height: float = 10) -> OCRLine:
    return OCRLine(text, 0.98, [[x, y], [x + width, y], [x + width, y + height], [x, y + height]])


class PPStructureTests(unittest.TestCase):
    def test_parses_layout_blocks_and_table_cells(self) -> None:
        result = parse_structure_result({
            "res": {
                "layout_det_res": {"boxes": [{"label": "table", "score": 0.91, "coordinate": [0, 0, 200, 100]}]},
                "table_res_list": [{
                    "cell_box_list": [
                        [0, 0, 50, 20], [50, 0, 100, 20], [100, 0, 150, 20], [150, 0, 200, 20],
                        [0, 20, 50, 40], [50, 20, 100, 40], [100, 20, 150, 40], [150, 20, 200, 40],
                    ],
                    "table_ocr_pred": {"rec_texts": [], "rec_boxes": []},
                }],
            }
        }, 200, 100)

        self.assertEqual(len(result.blocks), 1)
        self.assertEqual(result.blocks[0].label, "table")
        self.assertEqual(len(result.tables), 1)
        self.assertEqual(len(result.tables[0].cells), 8)
        self.assertEqual(result.reading_order, [0])

    def test_unifies_production_ocr_lines_with_structure_cells(self) -> None:
        structure = parse_structure_result({
            "res": {
                "layout_det_res": {"boxes": [{"label": "table", "score": 0.95, "coordinate": [0, 0, 200, 100]}]},
                "table_res_list": [{
                    "cell_box_list": [
                        [0, 0, 50, 20], [50, 0, 100, 20], [100, 0, 150, 20], [150, 0, 200, 20],
                        [0, 20, 50, 40], [50, 20, 100, 40], [100, 20, 150, 40], [150, 20, 200, 40],
                    ],
                    "table_ocr_pred": {"rec_texts": [], "rec_boxes": []},
                }],
            }
        }, 200, 100)
        lines = [
            text_line("Code", 5, 5), text_line("Qty", 55, 5), text_line("Description", 105, 5), text_line("Amount", 155, 5),
            text_line("A-1", 5, 25), text_line("2", 55, 25), text_line("Widget", 105, 25), text_line("10", 155, 25),
        ]

        schema = FieldSchema(tuple(FieldDefinition(name) for name in ("Code", "Qty", "Description", "Amount")), "test")
        result = extract_with_structure(lines, structure, schema=schema)

        self.assertEqual(result.layout_mode, "pp_structure")
        self.assertEqual(len(result.structure.tables[0].cells[0].line_indices), 1)
        self.assertTrue(any(field.label == "Code" for field in result.fields))
        self.assertTrue(any(field.value == ["A-1"] for field in result.fields))


if __name__ == "__main__":
    unittest.main()
