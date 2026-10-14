import json
import unittest
from ast import parse
from pathlib import Path

NOTEBOOK = Path(__file__).parents[1] / "Breast Cancer Code.ipynb"


class NotebookQualityTests(unittest.TestCase):
    def test_notebook_code_is_parseable_and_has_no_outputs(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                parse("".join(cell.get("source", [])))
                self.assertEqual(cell.get("outputs", []), [])
                self.assertIsNone(cell.get("execution_count"))

    def test_notebook_does_not_use_machine_specific_absolute_paths(self) -> None:
        text = NOTEBOOK.read_text(encoding="utf-8")
        self.assertNotIn("/home/", text)
        self.assertNotIn("/Users/", text)


if __name__ == "__main__":
    unittest.main()
