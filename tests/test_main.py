import io
import unittest
from contextlib import redirect_stdout

from ocr_test.main import main


class MainTests(unittest.TestCase):
    def test_main_prints_ready_message(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            main()
        self.assertIn("ocr-test project is ready.", output.getvalue())


if __name__ == "__main__":
    unittest.main()

