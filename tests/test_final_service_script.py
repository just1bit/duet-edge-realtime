import subprocess
import unittest
from pathlib import Path


class FinalServiceScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[1]
        cls.service = cls.repo / "scripts/final_execution/service.sh"

    def run_script(self, script: Path, *args: str):
        return subprocess.run(
            ["bash", str(script), *args], cwd=self.repo,
            text=True, capture_output=True, check=False,
        )

    def test_public_service_interface_is_reduced_to_four_commands(self):
        help_result = self.run_script(self.service, "--help")
        self.assertEqual(help_result.returncode, 0)
        usage = help_result.stderr
        for command in ("start", "stop", "status", "test", "mode"):
            self.assertIn(f"service.sh {command}", usage)
        self.assertNotIn("model start", usage)
        self.assertNotIn("stream start", usage)
        self.assertNotIn("viewer start", usage)
        self.assertIn("--template", usage)
        self.assertIn("--full-check", usage)

        rejected = self.run_script(self.service, "model", "start")
        self.assertEqual(rejected.returncode, 2)

    def test_start_rejects_conflicting_run_selection_without_side_effects(self):
        result = self.run_script(
            self.service, "start",
            "--run", "outputs/run-example",
            "--template", "configs/example.json",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot be used together", result.stderr)

    def test_shell_is_valid_and_calibration_is_bash_32_safe(self):
        syntax = self.run_script(self.service, "--help")
        self.assertEqual(syntax.returncode, 0)
        checked = subprocess.run(
            ["bash", "-n", str(self.service)], cwd=self.repo,
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr)
        source = self.service.read_text(encoding="utf-8")
        self.assertIn("calibrate_run", source)
        self.assertNotIn("input_args[@]", source)
        self.assertNotIn("03_baseline.sh", source)
        self.assertIn("RUNTIME_SERVICE", source)

        mediapipe = self.repo / "scripts/final_execution/mediapipe.sh"
        checked = subprocess.run(
            ["bash", "-n", str(mediapipe)], cwd=self.repo,
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr)
        help_result = self.run_script(mediapipe, "--help")
        self.assertEqual(help_result.returncode, 0)
        for command in ("start", "status", "stop", "debug", "doctor"):
            self.assertIn(f"mediapipe.sh {command}", help_result.stderr)

    def test_final_package_contains_runtime_and_maintenance_capabilities(self):
        final_root = self.repo / "scripts/final_execution"
        required = (
            "common.sh",
            "runtime_service.sh",
            "mediapipe.sh",
            "lib/capture_stage.py",
            "lib/runtime_client.py",
            "lib/run.py",
            "lib/export_fixture.py",
        )
        for relative in required:
            self.assertTrue((final_root / relative).is_file(), relative)
        run_source = (final_root / "lib/run.py").read_text(encoding="utf-8")
        self.assertIn("def command_report", run_source)

if __name__ == "__main__":
    unittest.main()
