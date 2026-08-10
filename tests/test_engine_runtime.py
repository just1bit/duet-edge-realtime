import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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

    def test_nondefault_sampling_rejects_engine_that_silently_ignores_kwargs(self):
        class BaselineDiffusion:
            def ddim_sample(self, shape, cond, **kwargs):
                sampling_timesteps, eta = 50, 1
                return shape, cond, sampling_timesteps, eta

        backend = CudaDuetEdgeBackend("model.pt", ".", sampling_steps=25)
        backend.edge = SimpleNamespace(diffusion=BaselineDiffusion())
        with self.assertRaisesRegex(RuntimeError, "does not honor configurable DDIM"):
            backend._validate_sampling_api()

    def test_nondefault_sampling_accepts_engine_that_consumes_options(self):
        class ConfigurableDiffusion:
            def ddim_sample(self, shape, cond, **kwargs):
                sampling_timesteps = kwargs.pop("sampling_timesteps", 50)
                eta = kwargs.pop("eta", 1.0)
                return shape, cond, sampling_timesteps, eta

        backend = CudaDuetEdgeBackend("model.pt", ".", sampling_steps=25)
        backend.edge = SimpleNamespace(diffusion=ConfigurableDiffusion())
        backend._validate_sampling_api()


if __name__ == "__main__":
    unittest.main()
