import subprocess
import unittest
from pathlib import Path


class V2ServiceScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[1]
        cls.service = cls.repo / "scripts/v2_execution/service.sh"

    def test_usage_rejects_unsupported_command_shapes(self):
        cases = (
            (),
            ("unknown",),
            ("model",),
            ("model", "status"),
        )
        for args in cases:
            with self.subTest(args=args):
                result = subprocess.run(
                    ["bash", str(self.service), *args],
                    cwd=self.repo,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("Usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
