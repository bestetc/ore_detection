import unittest
from pathlib import Path


class TestUiRunScripts(unittest.TestCase):
    def test_windows_cmd_script_runs_python313_backend_ui(self):
        script = Path("run_ui.cmd")
        self.assertTrue(script.exists())
        content = script.read_text(encoding="utf-8")
        self.assertIn(".venv\\Scripts\\python.exe", content)
        self.assertIn("py -3.13", content)
        self.assertIn("scripts\\run_backend_ui.py", content)
        self.assertIn("--host", content)
        self.assertIn("--port", content)

    def test_posix_script_runs_python313_backend_ui(self):
        script = Path("run_ui.sh")
        self.assertTrue(script.exists())
        content = script.read_text(encoding="utf-8")
        self.assertIn(".venv/bin/python", content)
        self.assertIn("python3.13", content)
        self.assertIn("py -3.13", content)
        self.assertIn("scripts/run_backend_ui.py", content)
        self.assertIn('PYTHONPATH="src', content)

    def test_windows_cpu_setup_script_installs_cpu_torch(self):
        script = Path("setup_ui_cpu.cmd")
        self.assertTrue(script.exists())
        content = script.read_text(encoding="utf-8")
        self.assertIn("py -3.13 -m venv .venv", content)
        self.assertIn("pip install -r requirements.txt", content)
        self.assertIn("pip install torch --index-url https://download.pytorch.org/whl/cpu", content)
        self.assertIn("run_ui.cmd", content)

    def test_posix_cpu_setup_script_installs_cpu_torch(self):
        script = Path("setup_ui_cpu.sh")
        self.assertTrue(script.exists())
        content = script.read_text(encoding="utf-8")
        self.assertIn("python3.13", content)
        self.assertIn("pip install -r requirements.txt", content)
        self.assertIn("pip install torch --index-url https://download.pytorch.org/whl/cpu", content)
        self.assertIn("pip install torch", content)
        self.assertIn("./run_ui.sh", content)

    def test_runtime_requirements_include_numpy_for_ui_arrays(self):
        requirements = Path("requirements.txt").read_text(encoding="utf-8").lower()

        self.assertIn("numpy", requirements)


if __name__ == "__main__":
    unittest.main()
