import tempfile
import unittest
from pathlib import Path

from duet_edge_realtime.backends.duet_edge import CudaDuetEdgeBackend


class EngineRuntimeTests(unittest.TestCase):
    def test_runtime_layout_accepts_required_model_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "duet-edge"
            (root / "model").mkdir(parents=True)
            for relative in ("EDGE.py", "model/diffusion.py", "vis.py"):
                (root / relative).write_text("# fixture\n")
            backend = CudaDuetEdgeBackend(Path(temp) / "model.pt", root)
            backend._validate_runtime_layout()

    def test_runtime_layout_reports_missing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "duet-edge"
            root.mkdir()
            backend = CudaDuetEdgeBackend(Path(temp) / "model.pt", root)
            with self.assertRaisesRegex(FileNotFoundError, "missing"):
                backend._validate_runtime_layout()


if __name__ == "__main__":
    unittest.main()
