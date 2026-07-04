import unittest
from pathlib import Path


class TestUiRunScripts(unittest.TestCase):
    def test_windows_cmd_script_runs_python313_backend_ui(self):
        script = Path("run_ui.cmd")
        self.assertTrue(script.exists())
        content = script.read_text(encoding="utf-8")
        self.assertIn("py -3.13", content)
        self.assertIn("scripts\\run_backend_ui.py", content)
        self.assertIn("--host", content)
        self.assertIn("--port", content)

    def test_posix_script_runs_python313_backend_ui(self):
        script = Path("run_ui.sh")
        self.assertTrue(script.exists())
        content = script.read_text(encoding="utf-8")
        self.assertIn("py -3.13", content)
        self.assertIn("scripts/run_backend_ui.py", content)
        self.assertIn("PYTHONPATH=src", content)


if __name__ == "__main__":
    unittest.main()
