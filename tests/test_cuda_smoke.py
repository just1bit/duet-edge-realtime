import os
import unittest
from pathlib import Path

import numpy as np

from duet_edge_realtime.backends.duet_edge import CudaDuetEdgeBackend
from duet_edge_realtime.schemas import MotionWindow

from helpers import identity_motion


@unittest.skipUnless(os.environ.get("RUN_CUDA_TESTS") == "1", "set RUN_CUDA_TESTS=1 on a CUDA node")
class CudaSmokeTests(unittest.TestCase):
    def test_single_window_is_finite_and_deterministic(self):
        checkpoint = os.environ["DUET_EDGE_CHECKPOINT"]
        engine = os.environ["DUET_EDGE_ROOT"]
        sampling_steps = 7
        backend = CudaDuetEdgeBackend(
            checkpoint, engine, sampling_steps=sampling_steps
        )
        backend.warmup()
        try:
            model_predictions = backend.edge.diffusion.model_predictions
            prediction_calls = 0

            def counted_model_predictions(*args, **kwargs):
                nonlocal prediction_calls
                prediction_calls += 1
                return model_predictions(*args, **kwargs)

            backend.edge.diffusion.model_predictions = counted_model_predictions
            window = MotionWindow(1, 0, 150, 0.0, 123, identity_motion(150))
            first = backend.infer(window).motion
            self.assertEqual(prediction_calls, sampling_steps)
            second = backend.infer(window).motion
            self.assertEqual(prediction_calls, sampling_steps * 2)
            self.assertEqual(first.shape, (150,151))
            self.assertTrue(np.isfinite(first).all())
            np.testing.assert_allclose(first, second, atol=1e-6, rtol=1e-6)
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
